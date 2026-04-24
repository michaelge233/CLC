import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader
from scipy.ndimage import zoom
from scipy.interpolate import interp1d

seed = 324
gap_wl = 150
n_resolution = 401
n_pol_train = 100
wl_x = np.linspace(400,800,n_resolution)

nn_inputsize = 448
batchsize = 16
n_epoch = 10

print("Using torch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
np.random.seed(seed)
print("Using seed %d"%seed)

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride= 1,expansion = 1,downsample= None):
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out
        
class ResNet18(nn.Module):
    def __init__(self,img_channels=1,block=BasicBlock,num_classes = 1000):
        super(ResNet18, self).__init__()
        num_layers = 18
        layers = [2, 2, 2, 2]
        self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(2048*self.expansion, num_classes)
        
    def _make_layer(self,block,out_channels,blocks,stride = 1):
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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

def evaluate(model,X,y,loss_fn):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred,y).item()
        AE = torch.abs(y_pred-y)
        sep_MAE = np.zeros(4,dtype=np.float32)
        for i in range(4):
            sep_MAE[i] = torch.mean(AE[:,i*n_resolution:(i+1)*n_resolution]).item()
    return loss,sep_MAE
    
    
# in rad
def stokes_from_alphabeta(alpha,beta):
    result = np.zeros((3,alpha.shape[0]),dtype=np.float32)
    result[0] = 0.5*(np.cos(4*alpha-4*beta)+np.cos(4*alpha))
    result[1] = 0.5*(np.sin(4*alpha)-np.sin(4*alpha-4*beta))
    result[2] = -np.sin(4*alpha-2*beta)
    return result
    
class BeamSpliterCorrecter:
    def __init__(self,filename):
        with open(filename,"r") as fo:
            file_list = fo.readlines()
        result=[]
        for i in range(1,len(file_list)):
            temp = file_list[i].strip().split()
            if len(temp)!=4:
                continue
            result.append([float(temp[0]),
                           float(temp[1]),
                           float(temp[2])])
        result = np.array(result)
        self.f_p = interp1d(result[:,0],result[:,1])
        self.f_s = interp1d(result[:,0],result[:,2])
    
    def get_new_alpha(self,wl,alpha,mode="R"):
    
        temp = np.cos(2*alpha)
        symbol = temp/np.abs(temp)
        s_part = np.square(temp)
        p_part = 1-s_part

        if mode=="R":
            ref_intensity = self.f_s(wl)
            s_part = ref_intensity*s_part
            p_part = self.f_p(wl)*p_part

        else:
            ref_intensity = 100-self.f_s(wl)
            s_part = ref_intensity*s_part
            p_part = (100-self.f_p(wl))*p_part

        new_alpha = np.arccos(symbol*np.sqrt(s_part)/np.sqrt(s_part+p_part))/2
        new_intensity = (s_part+p_part)/ref_intensity

        return new_alpha,new_intensity
    
    def __call__(self,wl_x,spec,alpha,beta,lamp=1):
        if lamp == 1:
            new_alpha,new_intensity = self.get_new_alpha(wl_x,alpha,mode="T")
        else:
            new_alpha,new_intensity = self.get_new_alpha(wl_x,alpha,mode="R")
        S123 = stokes_from_alphabeta(new_alpha,beta*np.ones_like(new_alpha))
        result = np.zeros(4*n_resolution,dtype=np.float32)
        result[:wl_x.shape[0]] = spec*new_intensity
        for i in range(3):
            result[(i+1)*wl_x.shape[0]:(i+2)*wl_x.shape[0]] = result[:wl_x.shape[0]]*S123[i]
            
        return result.astype(np.float32)
        
class TwobeamTrain(Dataset):
    def __init__(self):
        print("Loading training set:")
        print("  Reading spectrums...")
        self.spec_correcter = BeamSpliterCorrecter("./5to5.txt")
        self.lamp1_spec_raw,self.lamp1_cwl = self.read_spectrum("./spectrum/lamp1_spectrums.npz")
        self.lamp2_spec_raw,self.lamp2_cwl = self.read_spectrum("./spectrum/lamp2_spectrums.npz")
        self.spec_norm = np.max(self.lamp1_spec_raw)
        self.lamp1_spec_raw = self.lamp1_spec_raw/self.spec_norm
        self.lamp2_spec_raw = self.lamp2_spec_raw/self.spec_norm
        
        print("  Reading lamp1 images...")
        sys.stdout.flush()
        self.lamp1_imgs,self.lamp1_spec = self.get_img_spec(1,self.lamp1_spec_raw)
        print("  Reading lamp2 images...")
        sys.stdout.flush()
        self.lamp2_imgs,self.lamp2_spec = self.get_img_spec(2,self.lamp2_spec_raw)
        print("  Final processing...")
        self.img_norm = np.mean(self.lamp1_imgs)
        self.lamp1_imgs = torch.from_numpy(self.lamp1_imgs/self.img_norm)
        self.lamp2_imgs = torch.from_numpy(self.lamp2_imgs/self.img_norm)
        self.lamp1_spec = torch.from_numpy(self.lamp1_spec)
        self.lamp2_spec = torch.from_numpy(self.lamp2_spec)
        
        # self.aug_idx[i][j] represents i-th lamp2 can augment with aug_idx[i]-th lamp1
        self.aug_idx = []
        self.length = (self.lamp1_spec.shape[0] + self.lamp2_spec.shape[0])*n_pol_train
        for i in range(self.lamp2_cwl.shape[0]):
            self.aug_idx.append(np.argwhere(np.abs(self.lamp1_cwl-self.lamp2_cwl[i])>=gap_wl).reshape(-1))
            self.length += n_pol_train*n_pol_train*self.aug_idx[-1].shape[0]
        
        self.i1_cut = self.lamp1_spec.shape[0]*n_pol_train
        self.i2_cut = self.i1_cut + self.lamp2_spec.shape[0]*n_pol_train
        # i3_cut is a 1d array, indicating lamp2 wl cut. 
        self.i3_cut = np.zeros(len(self.aug_idx),dtype=np.int64)
        for i in range(1,len(self.aug_idx)):
            self.i3_cut[i] = self.i3_cut[i-1] + n_pol_train*n_pol_train*len(self.aug_idx[i-1])
        print("  Finish loading training set")
        sys.stdout.flush()
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < self.i1_cut:
            wl_i,pol_i = divmod(idx,n_pol_train)
            return self.lamp1_imgs[wl_i,pol_i], self.lamp1_spec[wl_i,pol_i]
        if idx < self.i2_cut:
            wl_i,pol_i = divmod(idx-self.i1_cut,n_pol_train)
            return self.lamp2_imgs[wl_i,pol_i], self.lamp2_spec[wl_i,pol_i]
        
        i3 = idx-self.i2_cut
        l2_wl_i = np.searchsorted(self.i3_cut,i3,side='right') - 1
        l2_pol_i, l1_all_i = divmod(i3-self.i3_cut[l2_wl_i], len(self.aug_idx[l2_wl_i])*n_pol_train)
        l1_wl_i, l1_pol_i = divmod(l1_all_i, n_pol_train)
        
        img = self.lamp1_imgs[self.aug_idx[l2_wl_i][l1_wl_i],l1_pol_i] + self.lamp2_imgs[l2_wl_i, l2_pol_i]
        spec = self.lamp1_spec[self.aug_idx[l2_wl_i][l1_wl_i],l1_pol_i] + self.lamp2_spec[l2_wl_i, l2_pol_i]
        return img,spec
    
    def read_spectrum(self,filename):
        fo = np.load(filename)
        cwl = fo["arr_1"]
        raw_spec = fo["arr_0"]
        spec = np.zeros((raw_spec.shape[0],n_resolution))
        for i in range(raw_spec.shape[0]):
            spec[i] = np.interp(np.linspace(400,800,n_resolution),
                                np.linspace(400,800,raw_spec.shape[1]),
                                raw_spec[i])
        return spec,cwl
    
    def read_img(self,filename):
        fo = np.load(filename)
        raw_imgs = fo["arr_0"]
        exp_array = fo["arr_1"]
        alpha_array = fo["arr_2"]/180*np.pi
        beta_array = fo["arr_3"]/180*np.pi
        imgs = np.zeros((exp_array.shape[0],1,nn_inputsize,nn_inputsize),
                        dtype=np.float32)
        for i in range(exp_array.shape[0]):
            imgs[i,0] = zoom(raw_imgs[i]/exp_array[i],
                            (nn_inputsize/raw_imgs.shape[1],
                             nn_inputsize/raw_imgs.shape[2])).astype(np.float32)
            
        return imgs,alpha_array,beta_array
    
    def get_img_spec(self,lamp,raw_spec):
        file_list = os.listdir("./single/")
        target_files = []
        for i in range(len(file_list)):
            if file_list[i][-4:] == ".npz" and file_list[i].split("_")[0] == "lamp%d"%lamp:
                target_files.append(file_list[i])
        imgs = np.zeros((len(target_files),n_pol_train,
                           1,nn_inputsize,nn_inputsize),
                           dtype=np.float32)
        specs = np.zeros((len(target_files),n_pol_train,4*n_resolution),dtype=np.float32)
        for i in range(len(target_files)):
            imgs[i],alphas,betas = self.read_img("./single/"+target_files[i])
            for j in range(n_pol_train):
                specs[i,j] = self.spec_correcter(wl_x,raw_spec[i],alphas[j],betas[j],lamp=lamp)
            
        return imgs,specs

def get_valtest(training_set):
    print("Loading test and validation sets.")
    file_list = os.listdir("./mix/")
    target_list = []
    for i in range(len(file_list)):
        if file_list[i][-4:] == ".npz":
            target_list.append(file_list[i])
    all_imgs = []
    all_y = []
    for i in range(len(target_list)):
        temp = target_list[i][:-4].split("_")
        lamp1_spec = training_set.lamp1_spec_raw[np.argmin(np.abs(training_set.lamp1_cwl-float(temp[0])))]
        lamp2_spec = training_set.lamp2_spec_raw[np.argmin(np.abs(training_set.lamp2_cwl-float(temp[1])))]
        
        fo = np.load("./mix/"+target_list[i])
        raw_imgs = fo["arr_0"]
        exp_time = fo["arr_1"]
        alpha1_array = fo["arr_2"]/180*np.pi
        alpha2_array = fo["arr_3"]/180*np.pi
        beta_array = fo["arr_4"]/180*np.pi
        imgs = np.zeros((raw_imgs.shape[0],1,nn_inputsize,nn_inputsize),dtype=np.float32)
        ys = np.zeros((raw_imgs.shape[0],4*n_resolution),dtype=np.float32)
        for j in range(raw_imgs.shape[0]):
            imgs[j,0] = zoom(raw_imgs[j]/exp_time[j],
                            (nn_inputsize/raw_imgs.shape[1],
                             nn_inputsize/raw_imgs.shape[2])).astype(np.float32)
            ys[j] += training_set.spec_correcter(wl_x,lamp1_spec,alpha1_array[j],beta_array[j],lamp=1)
            ys[j] += training_set.spec_correcter(wl_x,lamp2_spec,alpha2_array[j],beta_array[j],lamp=2)
        all_imgs.append(imgs)
        all_y.append(ys)
            
    all_imgs = np.concatenate(all_imgs)/training_set.img_norm
    all_y = np.concatenate(all_y)
    
    shuffled_idx = np.arange(all_y.shape[0])
    np.random.shuffle(shuffled_idx)
    cut_i = all_y.shape[0]//2
    x_val = torch.from_numpy(all_imgs[shuffled_idx[:cut_i]]).to(device)
    x_test = torch.from_numpy(all_imgs[shuffled_idx[cut_i:]]).to(device)
    y_val = torch.from_numpy(all_y[shuffled_idx[:cut_i]]).to(device)
    y_test = torch.from_numpy(all_y[shuffled_idx[cut_i:]]).to(device)
    
    return x_val,y_val,x_test,y_test

def run_experiment(task_name,seed_list=[seed],batchsize=batchsize,n_epoch=n_epoch,save_dir="./ML_result/"):
    result_name = ["train_loss",
                   "val_loss","val_S0","val_S1","val_S2","val_S3",
                   "test_loss","test_S0","test_S1","test_S2","test_S3"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print("Running %s"%task_name)
    for seed in seed_list:
        print("Seed %d"%seed)
        sys.stdout.flush()
        torch.manual_seed(seed)
        train_loader = DataLoader(training_set,batch_size=batchsize, shuffle=True)
        model=ResNet18(num_classes=n_resolution*4).to(device)
        loss_fn=MSELoss()
        optimizer=Adam(model.parameters(),lr=0.0001)
        result=np.zeros((n_epoch,11),dtype=np.float32)
        for i in range(n_epoch):
            result[i,0] = train(model,train_loader,optimizer,loss_fn)
            result[i,1],result[i,2:6] = evaluate(model,x_val,y_val,loss_fn)
            result[i,6],result[i,7:] = evaluate(model,x_test,y_test,loss_fn)
            print(i,result)
            sys.stdout.flush()
            torch.save(model,save_dir+task_name+"_seed%d_epoch%d"%(seed,i+1) + ".pt")
        result = pd.DataFrame(result,columns=result_name)
        save_filename=save_dir+task_name+"_seed%d"%seed+".csv"
        result.to_csv(save_filename,index=False)
        
training_set = TwobeamTrain()
x_val,y_val,x_test,y_test = get_valtest(training_set)
run_experiment("Resnet18")
print("All finished!")
