import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
from scipy.interpolate import splprep, splev
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--n_epoch", type=int, default=5000)
parser.add_argument("--seed", type=int, default=324)
parser.add_argument("--task_name", default="UNet_small")
parser.add_argument("--start_model", default=None)
args = parser.parse_args()
save_dic = vars(args)

seed = args.seed
print("Using torch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
np.random.seed(seed)
torch.manual_seed(seed)
print("Using seed %d"%seed)
sys.stdout.flush()

batch_size = args.batch_size
n_epoch = args.n_epoch
num_workers = args.num_workers
task_name = args.task_name
HW = (args.img_size,args.img_size)
start_model = args.start_model

# ----------------------------------For dataset loading--------------------------------------------

def stokes_from_alphabeta(alpha,beta):
    # fast axis of HWP and QWP should be vertical
    # alpha: an array of HWP angels, in rad
    # beta: an array of QEP angels, in rad
    # return array(n,3), each row represent [S1/S0,S2/S0,S3/S0]
    result = np.zeros((alpha.shape[0],3),dtype=np.float32)
    result[:,0] = 0.5*(np.cos(4*alpha-4*beta)+np.cos(4*alpha))
    result[:,1] = 0.5*(np.sin(4*alpha)-np.sin(4*alpha-4*beta))
    result[:,2] = -np.sin(4*alpha-2*beta)
    return result

def gaussian_random_field(shape, mean=0.0, std=1.0, alpha=2.5):
    """
    Generate a 2D Gaussian random field with specified mean, std, and spectral exponent alpha.

    Parameters:
        shape (tuple): Shape of the output field as (ny, nx).
        mean (float): Mean of the output field.
        std (float): Standard deviation of the output field.
        alpha (float): Controls the smoothness (higher = smoother).
        seed (int or None): Random seed for reproducibility.

    Returns:
        field (2D np.ndarray): Real-valued Gaussian random field.
    """
    ny, nx = shape
    kx = np.fft.fftfreq(nx).reshape(1, nx)
    ky = np.fft.fftfreq(ny).reshape(ny, 1)
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1e-10  # prevent division by zero at origin
    amplitude = 1.0 / (k_squared + 1.0)**(alpha / 2.0)
    noise_real = np.random.normal(size=(ny, nx))
    noise_imag = np.random.normal(size=(ny, nx))
    noise = noise_real + 1j * noise_imag
    fourier_field = amplitude * noise
    field = np.fft.ifft2(fourier_field).real

    field -= np.mean(field)
    field /= np.std(field)
    field = field * std + mean

    return field


class ShapeGen:
    def __init__(self,pol_size=1000,HW=HW,cut_region=(0,1216,0,1936),gm=2.2):
        self.pol_size = pol_size
        self.HW = HW
        self.cut_region = cut_region
        self.gm = 2.2
        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.norm_fact = 0
        self.rgb_train = self.load_rgb_train()
        self.S = self.get_S()
        
    def load_rgb_train(self,paths=["./red/","./green/","./blue/"]):
        print("Loading RGB train")
        rgb_train = np.zeros((1000,3,self.HW[0],self.HW[1]),dtype=np.float32)
        for i in range(3):
            for j in range(10):
                fo = np.load(paths[i]+"data%d.npz"%j)
                cur_imgs = fo["arr_0"][:,self.cut_region[0]:self.cut_region[1],
                                         self.cut_region[2]:self.cut_region[3]]
                cur_exp = fo["arr_1"]
                for n in range(100):
                    rgb_train[j*100+n,i,:,:] = zoom(cur_imgs[n]/cur_exp[n],
                                                (self.HW[0]/cur_imgs.shape[1],self.HW[1]/cur_imgs.shape[2])).astype(np.float32)
        self.norm_fact = np.max(np.sum(rgb_train,3))
        rgb_train = torch.from_numpy(rgb_train/self.norm_fact)
        print("This dataset is normalized by a factor of %f"%self.norm_fact)
        return rgb_train

    def get_S(self):
        fo = np.load("./green/random_parameters.npz")
        alpha_array = fo["arr_0"].astype(np.float32)/180*np.pi
        beta_array = fo["arr_1"].astype(np.float32)/180*np.pi
        S = stokes_from_alphabeta(alpha_array,beta_array)
        S = torch.from_numpy(S)
        return S

    def rect_mask(self):
        # Create a grid of coordinates
        H = self.HW[0]
        W = self.HW[1]
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        rect_h = np.random.randint(H // 10, H//2)
        rect_w = np.random.randint(W // 10, W//2)
        center_y = np.random.randint(rect_h//2, H - rect_h//2)
        center_x = np.random.randint(rect_w//2, W - rect_w//2)
        angle = np.deg2rad(np.random.uniform(0, 360))
        Yc = Y - center_y
        Xc = X - center_x
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        X_rot = cos_a * Xc - sin_a * Yc
        Y_rot = sin_a * Xc + cos_a * Yc
        mask = (np.abs(X_rot) <= rect_w / 2) & (np.abs(Y_rot) <= rect_h / 2)
        output = torch.zeros((H, W), dtype=torch.uint8)
        output[mask] = 1

        return output
    
    def polygon_mask(self, num_vertices=6, scale_range=(0.1, 0.4), aspect_ratio_range=(0.5, 2.0)):
        """
        Generate a binary mask with a regular polygon.

        Parameters:
        - H, W: Output mask height and width.
        - num_vertices: Number of sides (>=3).
        - scale_range: Range for polygon radius relative to image size.
        - aspect_ratio_range: Min/max height-to-width ratio for stretching.

        Returns:
        - mask: Binary (H, W) numpy array with polygon area = 1, else 0.
        """
        H = self.HW[0]
        W = self.HW[1]
        cx, cy = W / 2, H / 2  # center of image
        # Radius (scaling factor relative to image size)
        scale = np.random.uniform(*scale_range) * min(H, W)
        # Regular angles around circle
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        # Random rotation
        rotation = np.random.uniform(0, 2 * np.pi)
        angles += rotation
        # Aspect ratio stretch
        aspect_ratio = np.random.uniform(*aspect_ratio_range)
        sx = 1.0
        sy = aspect_ratio
        # Coordinates of the regular polygon
        x = scale * np.cos(angles) * sx + cx
        y = scale * np.sin(angles) * sy + cy
        polygon = np.stack([x, y], axis=-1)
        # Draw polygon on a blank image
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)

        return torch.from_numpy(mask)

    def ellipse_mask(self, scale_range=(0.1, 0.5)):
        """
        Generate a binary mask with a randomly sized and rotated ellipse.

        Parameters:
        - H, W: Height and width of the output array.
        - scale_range: Tuple for relative size of the ellipse's axes (min, max) as a fraction of image size.

        Returns:
        - mask: (H, W) numpy array, values are 1 inside the ellipse, 0 outside.
        """
        H = self.HW[0]
        W = self.HW[1]
        center_x = np.random.randint(W // 4, 3 * W // 4)
        center_y = np.random.randint(H // 4, 3 * H // 4)
        center = (center_x, center_y)
        axis_x = int(np.random.uniform(*scale_range) * W)
        axis_y = int(np.random.uniform(*scale_range) * H)
        angle = np.random.uniform(0, 360)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, center, (axis_x, axis_y), angle,
                    startAngle=0, endAngle=360, color=1, thickness=-1)
        return torch.from_numpy(mask)

    def letter_mask(self,n=None):
        image_size = self.HW
        if n is None:
            n = np.random.randint(0,26)
        letter = self.letters[n]
        temp_size = int(max(image_size) * 2)
        img = Image.new("L", (temp_size, temp_size), 0)
        draw = ImageDraw.Draw(img)
        # Random font size
        font_size = np.random.randint(temp_size//4, temp_size // 2)
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except OSError:
            raise RuntimeError("Arial Black font not found. Ensure 'arialbd.ttf' is available.")
        # Draw the text centered

        text_left, text_top, text_right, text_bottom = draw.textbbox((0,0), letter, font=font)
        text_width, text_height = (text_right - text_left, text_bottom - text_top)
        #text_width, text_height = draw.textsize(letter, font=font)
        x = (temp_size - text_width) // 2
        y = (temp_size - text_height) // 2
        draw.text((x, y), letter, fill=255, font=font)
        # Apply random x and y zoom (scaling)
        zoom_x = np.random.uniform(0.5, 2.0)
        zoom_y = np.random.uniform(0.5, 2.0)
        new_size = (int(temp_size * zoom_x), int(temp_size * zoom_y))
        zoomed_img = img.resize(new_size, resample=Image.BILINEAR)
        # Rotate the image
        angle = np.random.uniform(-20, 20)
        rotated_img = zoomed_img.rotate(angle, expand=True, fillcolor=0)
        # Center-crop to the output size
        cx, cy = rotated_img.size[0] // 2, rotated_img.size[1] // 2
        left = cx - image_size[1] // 2
        upper = cy - image_size[0] // 2
        cropped_img = rotated_img.crop((left, upper, left + image_size[1], upper + image_size[0]))
        # Convert to binary NumPy array
        arr = np.array(cropped_img)
        binary_arr = (arr > 0).astype(np.uint8)

        return torch.from_numpy(binary_arr)
    
    def draw_curve(self, X, Y, num_control_points=6, noise_center=2.539e-5, noise_std=8.1e-5, thickness=8):
        x = np.linspace(0, self.HW[1] - 1, num_control_points)
        y = np.random.randint(0, self.HW[0], size=num_control_points)
        tck, _ = splprep([x, y], s=0)
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)
        curve_mask = np.zeros(self.HW, dtype=np.uint8)
        for i in range(1, len(x_fine)):
            pt1 = (int(x_fine[i - 1]), int(y_fine[i - 1]))
            pt2 = (int(x_fine[i]), int(y_fine[i]))
            cv2.line(curve_mask, pt1, pt2, color=255, thickness=thickness)
        curve_indices = np.argwhere(curve_mask == 255)
        noise = gaussian_random_field(self.HW,mean=noise_center,std=noise_std,alpha=1000.0)
        X[0,curve_indices[:,0],curve_indices[:,1]] = torch.from_numpy(noise[curve_indices[:,0],curve_indices[:,1]].astype(np.float32))
        Y[:,curve_indices[:,0],curve_indices[:,1]] = 0

        return X, Y


    def __call__(self, gentype, c1=None, c2=None, p1=None, p2=None, has_curve=None):
        if c1 is None:
            c1 = torch.rand(3)
        if c2 is None:
            c2 = torch.rand(3)
        if p1 is None:
            p1 = np.random.randint(0,1000)
        if p2 is None:
            p2 = np.random.randint(0,1000)
        if has_curve == None:
            has_curve = np.random.rand() > 0.5

        Y = torch.zeros((6,self.HW[0],self.HW[1]),dtype=torch.float32)
        ONEs = torch.ones((3,self.HW[0],self.HW[1]),dtype=torch.float32)
        with torch.no_grad():
            c1_gm = torch.pow(c1,self.gm)
            c2_gm = torch.pow(c2,self.gm)
            if gentype == 0:
                # a single color and polarization
                X = torch.sum(self.rgb_train[p1]*c1_gm[:,None,None],0)
                Y[:3] = ONEs*c1[:,None,None]
                Y[3:] = ONEs*self.S[p1][:,None,None]

            elif gentype == 1:
                # a single color and 2 polarizations
                rect_mask = self.rect_mask()
                X = torch.sum(self.rgb_train[p1]*c1_gm[:,None,None],0)*rect_mask
                X += torch.sum(self.rgb_train[p2]*c1_gm[:,None,None],0)*torch.logical_not(rect_mask)
                
                Y[3:] = ONEs*self.S[p1][:,None,None]*rect_mask[None,:,:]
                Y[3:] += ONEs*self.S[p1][:,None,None]*torch.logical_not(rect_mask[None,:,:])
                Y[:3] = ONEs*c1[:,None,None]

            elif gentype == 2:
                # Shape, 2 polarization
                rect_mask = self.rect_mask()
                s = np.random.rand()
                if s>=0.9:
                    cur_mask = self.ellipse_mask()
                elif s>=0.5:
                    cur_mask = self.polygon_mask(num_vertices=int((s-0.5)*10)+3)
                else:
                    cur_mask = self.letter_mask()

                mask_overlap = torch.logical_and(rect_mask,cur_mask)
                no_mask = torch.logical_not(torch.logical_or(rect_mask,cur_mask))
                color_only = torch.logical_xor(cur_mask,mask_overlap)
                p_only = torch.logical_xor(rect_mask,mask_overlap)

                X = torch.sum(self.rgb_train[p1]*c1_gm[:,None,None],0)*no_mask
                X += torch.sum(self.rgb_train[p1]*c2_gm[:,None,None],0)*color_only
                X += torch.sum(self.rgb_train[p2]*c1_gm[:,None,None],0)*p_only
                X += torch.sum(self.rgb_train[p2]*c2_gm[:,None,None],0)*mask_overlap

                Y[3:] = ONEs*self.S[p1][:,None,None]*torch.logical_not(rect_mask[None,:,:])
                Y[3:] += ONEs*self.S[p2][:,None,None]*rect_mask[None,:,:]
                Y[:3] = ONEs*c1[:,None,None]*torch.logical_not(cur_mask[None,:,:])
                Y[:3] += ONEs*c2[:,None,None]*cur_mask[None,:,:]

            X = X[None,:,:]
            if has_curve:
                X, Y = self.draw_curve(X, Y)

        return X,Y

class ShapeTrain(Dataset):
    def __init__(self,generator):
        self.num0, self.num1, self.num2 = 100, 100, 800
        self.generator = generator
        self.length = self.num0 + self.num1 + self.num2 

    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        if idx<self.num0:
            return self.generator(0)
        elif idx<self.num1:
            return self.generator(1)
        else:
            return self.generator(2)
    
class ShapeVal(Dataset):
    def __init__(self,generator,length=100):
        self.generator = generator
        self.length = length
        self.X = torch.zeros((self.length,1,generator.HW[0],generator.HW[1]),dtype=torch.float32)
        self.Y = torch.zeros((self.length,6,generator.HW[0],generator.HW[1]),dtype=torch.float32)

        for i in range(length):
            self.X[i], self.Y[i] = generator(2)

    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]
    


# ------------------------------------  UNet ----------------------------------------------
# Adapted from https://github.com/milesial/Pytorch-UNet/
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# -------------------------------------Training/validate functions---------------------------------------------------
def train(model,dataloader,optimizer,loss_fn):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        cur_x = batch[0].to(device)
        cur_y = batch[1].to(device)
        optimizer.zero_grad()
        y_pred = model(cur_x)
        loss = loss_fn(y_pred, cur_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*cur_y.shape[0]
    return epoch_loss/len(dataloader)

def evaluate(model,dataloader,loss_fn):
    model.eval()
    sep_MAE = np.zeros(4,dtype=np.float32)
    loss = 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred,y).item()*y.shape[0]
            AE = torch.abs(y_pred-y)
            
            sep_MAE[0] += torch.mean(AE[:,:3]).item()*y.shape[0]
            for i in range(1,4):
                sep_MAE[i] += torch.mean(AE[:,i+2]).item()*y.shape[0]

    loss = loss/len(dataloader)
    sep_MAE = sep_MAE/len(dataloader)

    return loss,sep_MAE

def write_y_pred(model,dataloader,path):
    model.eval()
    y_pred_all = []
    y_all = []
    X_all = []
    cur_i = 0
    n_file = 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(X)
            y_pred = y_pred.detach().to("cpu").numpy()
            y = y.detach().to("cpu").numpy()
            X = X.detach().to("cpu").numpy()
            y_pred_all.append(y_pred)
            y_all.append(y)
            X_all.append(X)
            cur_i += 1
            if cur_i==9:
                y_pred_all = np.concatenate(y_pred_all)
                y_all = np.concatenate(y_all)
                X_all = np.concatenate(X_all)
                np.savez_compressed(path+"y%d"%n_file,y_pred_all,y_all,X_all)
                y_pred_all = []
                y_all = []
                X_all = []
                cur_i = 0
                n_file +=1
        if cur_i != 9:
            y_pred_all = np.concatenate(y_pred_all)
            y_all = np.concatenate(y_all)
            X_all = np.concatenate(X_all)
            np.savez_compressed(path+"y%d"%n_file,y_pred_all,y_all,X_all)

def run_experiment(task_name,save_dir="./ML_result/"):
    result_name = ["train_loss",
                   "val_loss","val_S0","val_S1","val_S2","val_S3",]
    os.makedirs(save_dir,exist_ok=True) 
    print("Running %s"%task_name)
    if start_model is None:
        model = UNet(1,6).to(device)
        print("Weights initialized randomly")
    else:
        model = torch.load(start_model,map_location=device).to(device)
        print("Weights initialized from %s"%start_model)
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(),lr=0.0001)
    result = np.zeros((n_epoch//10+1,len(result_name)),dtype=np.float32)

    print("Loading dataset")
    sys.stdout.flush()
    generator = ShapeGen(cut_region=(50,1150,450,1550))
    train_set = ShapeTrain(generator)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_set = ShapeVal(generator)
    val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    save_dic["norm_factor"] = float(generator.norm_fact)
    save_dic["cut_region"] = list(generator.cut_region)
    with open(save_dir+task_name+"_seed%d.json"%seed, "w") as f:
        json.dump(save_dic,f,indent=4)

    print("Start training")
    for i in range(0,n_epoch):
        result[i//10,0]=train(model,train_loader,optimizer,loss_fn)
        if i%10==0:
            result[i//10,1], result[i//10,2:] = evaluate(model,val_loader,loss_fn)
            print(i,result[i//10])
            sys.stdout.flush()
            if i == 0:
                continue
            if np.min(result[:i//10,1])>result[i//10,1]:
                torch.save(model,save_dir+task_name+"_seed%d"%seed+".pt")
    
    model = torch.load(save_dir+task_name+"_seed%d"%seed+".pt")
    write_y_pred(model,val_loader,save_dir+task_name+"_seed%d"%seed)
    result=pd.DataFrame(result,columns=result_name)
    save_filename=save_dir+task_name+"_seed%d"%seed+".csv"
    result.to_csv(save_filename,index=False)

if __name__ == "__main__":
    run_experiment(task_name)
    """generator = ShapeGen(cut_region=(50,1150,450,1550))
    train_set = ShapeTrain(generator)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    for batch in train_loader:
        X = batch[0].numpy()
        Y = batch[1].numpy()
        break
    np.savez_compressed("batch_example.npz",X,Y)"""
    
    print("All finished...")


