import os
import sys
import numpy as np
import torch
from scipy.ndimage import zoom
from math import pi
from torch import nn
import pandas as pd
from torch.nn import MSELoss
from torch.optim import Adam

print("Using torch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
sys.stdout.flush()

interesting_region=np.load("./interesting_region.npy")
fo=np.load("./random_parameters.npz")
wl_array=fo["arr_0"]
slit_array=fo["arr_1"]
alpha_array=fo["arr_2"]
beta_array=fo["arr_3"]

dataset_size=5000
split_size=[4500,250,250]
seed=715
wl_resolution = 60

ori_size=np.load("./data0.npz")["arr_0"][0].shape
active_y=min(interesting_region[1,0],ori_size[0])-max(0,interesting_region[0,0])
active_x=min(interesting_region[1,1],ori_size[1])-max(0,interesting_region[0,1])

def zero_outside(img_array):
    mid=np.mean(interesting_region,0)
    a2=(interesting_region[1,1]-interesting_region[0,1])/2
    a2=a2**2
    b2=(interesting_region[1,0]-interesting_region[0,0])/2
    b2=b2**2
    if_keep=np.zeros((img_array.shape[1],img_array.shape[2]),dtype=bool)
    black=np.zeros((img_array.shape[1],img_array.shape[2]),dtype=img_array.dtype)
    for i in range(img_array.shape[1]):
        for j in range(img_array.shape[2]):
            if (i-mid[0])**2/b2 + (j-mid[1])**2/a2 <=1:
                if_keep[i,j]=True
    for i in range(img_array.shape[0]):
        img_array[i]=np.where(if_keep,img_array[i],black)

def stokes_from_alphabeta(alpha,beta):
    alpha=alpha/180*pi
    beta=beta/180*pi
    result=np.zeros(3,dtype=np.float32)
    result[0]=0.5*(np.cos(4*alpha-4*beta)+np.cos(4*alpha))
    result[1]=0.5*(np.sin(4*alpha)-np.sin(4*alpha-4*beta))
    result[2]=-np.sin(4*alpha-2*beta)
    return result
    
# Adapted from https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
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
    def __init__(self,img_channels=1,block=BasicBlock,num_classes = 240):
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
        
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
        
all_imgs=np.zeros((dataset_size,1,224,224),dtype=np.float32)
fo_p=np.load("./random_parameters.npz")
alpha_array=fo_p["arr_2"]
beta_array=fo_p["arr_3"]

print("Loading dataset")
sys.stdout.flush()
for i in range(dataset_size//100):
    fo=np.load("./data%d.npz"%i)
    temp0=fo["arr_0"]
    zero_outside(temp0)
    temp0=temp0[:,interesting_region[0,0]:interesting_region[1,0],interesting_region[0,1]:interesting_region[1,1]]
    temp1=fo["arr_1"]
    for j in range(100):
        all_imgs[i*100+j,0] = zoom(temp0[j]*temp1[j],(224/active_y,224/active_x))
all_imgs=all_imgs/np.max(all_imgs)

all_y=np.zeros((dataset_size,wl_resolution*4),dtype=np.float32)
for i in range(dataset_size//100):
    fo=np.load("./specturm_collab/data%d.npz"%i)
    temp0=fo["arr_0"]
    temp1=fo["arr_1"]
    if i>0 and i!=2 and i!=36:
        fo=np.load("./specturm_collab/data%d.npz"%(i-1))
        temp0=temp0-fo["arr_0"]
        
    for j in range(100):
        all_y[i*100+j,:wl_resolution] = zoom(temp0[j]/temp1[j]/60000000,wl_resolution/712)
        
        stokes = stokes_from_alphabeta(alpha_array[i*100+j],beta_array[i*100+j])
        all_y[i*100+j,wl_resolution:2*wl_resolution] = stokes[0]*all_y[i*100+j,:wl_resolution]
        all_y[i*100+j,2*wl_resolution:3*wl_resolution] = stokes[1]*all_y[i*100+j,:wl_resolution]
        all_y[i*100+j,3*wl_resolution:] = stokes[2]*all_y[i*100+j,:wl_resolution]

        
np.random.seed(seed)
shuffled_idx=np.arange(dataset_size)
np.random.shuffle(shuffled_idx)
all_imgs=all_imgs[shuffled_idx]
all_y=all_y[shuffled_idx]

x_train=all_imgs[:split_size[0]]
y_train=all_y[:split_size[0]]
x_val=all_imgs[split_size[0]:split_size[0]+split_size[1]]
y_val=all_y[split_size[0]:split_size[0]+split_size[1]]
x_test=all_imgs[split_size[0]+split_size[1]:]
y_test=all_y[split_size[0]+split_size[1]:]

x_train=torch.from_numpy(x_train).to(device)
y_train=torch.from_numpy(y_train).to(device)
x_val=torch.from_numpy(x_val).to(device)
y_val=torch.from_numpy(y_val).to(device)
x_test=torch.from_numpy(x_test).to(device)
y_test=torch.from_numpy(y_test).to(device)

def train(model,X,y,optimizer,loss_fn,batchsize=16):
    idx=np.arange(X.shape[0])
    np.random.shuffle(idx)
    
    epoch_loss = 0
    model.train()
    for i in range(0,X.shape[0],batchsize):
        if i+batchsize>=X.shape[0]:
            cur_idx=idx[i:]
        else:
            cur_idx=idx[i:i+batchsize]
            
        optimizer.zero_grad()
        y_pred = model(X[cur_idx])
        loss = loss_fn(y_pred, y[cur_idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*cur_idx.shape[0]
    return epoch_loss/X.shape[0]
    
def evaluate(model,X,y,loss_fn,batchsize=16):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for i in range(0,X.shape[0],batchsize):
            i_start=i
            if i+batchsize>=X.shape[0]:
                i_end=X.shape[0]-1
            else:
                i_end=i+batchsize
                
            y_pred=model(X[i_start:i_end])
            loss = loss_fn(y_pred, y[i_start:i_end])
            epoch_loss += loss.item()*(i_end-i_start)
            
        epoch_loss=epoch_loss/X.shape[0]
    return epoch_loss

def run_cnn_exp(task_name,seed_list=[715],n_epoch=5001,save_dir="./ML_result/"):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    score=[]
    print("Running %s"%task_name)
    for seed in seed_list:
        torch.manual_seed(seed)
        model=ResNet18()
        model = model.to(device)
        loss_fn=MSELoss()
        optimizer=Adam(model.parameters(),lr=0.0001)
        result=np.zeros((n_epoch//10+1,3),dtype=np.float32)
        
        first_stage=round(n_epoch*0.5)
        for i in range(first_stage):
            result[i//10,0]=train(model,x_train,y_train,optimizer,loss_fn)
            if i%10==0:
                result[i//10,1]=evaluate(model,x_val,y_val,loss_fn)
                result[i//10,2]=evaluate(model,x_test,y_test,loss_fn)
                print(i,result[i//10])
                sys.stdout.flush()
        
        torch.save(model,save_dir+task_name+"_seed%d"%seed+".pt")
        #optimizer.param_groups[0]['lr'] = 0.0005
        for i in range(first_stage,n_epoch):
            result[i//10,0]=train(model,x_train,y_train,optimizer,loss_fn)
            if i%10==0:
                result[i//10,1]=evaluate(model,x_val,y_val,loss_fn)
                result[i//10,2]=evaluate(model,x_test,y_test,loss_fn)
                print(i,result[i//10])
                sys.stdout.flush()
                if np.min(result[:i//10,1])>result[i//10,1]:
                    torch.save(model,save_dir+task_name+"_seed%d"%seed+".pt")

        best_idx=np.argmin(result[:,1])
        score.append(result[best_idx,2])
        result=pd.DataFrame(result,columns=["train_loss","validation_loss","test_loss"])
        save_filename=save_dir+task_name+"_seed%d"%seed+".csv"
        result.to_csv(save_filename,index=False)
    return score

print("Start training...")
run_cnn_exp("ResNet_spec_lr-4",seed_list=[seed])
print("All finished...")

