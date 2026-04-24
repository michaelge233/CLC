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

seed = 324
nn_inputsize = 448
batchsize = 16
n_epoch = 2000
n_resolution = 401

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

def train(model,X,y,optimizer,loss_fn,batchsize=batchsize):
    idx=np.arange(X.shape[0])
    np.random.shuffle(idx)
    
    epoch_loss = 0
    model.train()
    for i in range(0,X.shape[0],batchsize):
        if i+batchsize>=X.shape[0]:
            cur_idx=idx[i:]
        else:
            cur_idx=idx[i:i+batchsize]
            
        cur_scale = np.random.rand() + 0.5
        optimizer.zero_grad()
        y_pred = model(X[cur_idx].to(device)*cur_scale)
        loss = loss_fn(y_pred, y[cur_idx]*cur_scale)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*cur_idx.shape[0]
        
    return epoch_loss/X.shape[0]

def evaluate(model,X,y,loss_fn,batchsize=batchsize):
    idx=np.arange(X.shape[0])
    model.eval()
    loss = 0
    L1_re = np.zeros(4,dtype=np.float32)
    all_y_pred = np.zeros((y.shape[0],y.shape[1]),dtype=np.float32)
    with torch.no_grad():
        for i in range(0,X.shape[0],batchsize):
            if i+batchsize>=X.shape[0]:
                cur_idx=idx[i:]
            else:
                cur_idx=idx[i:i+batchsize]
            y_pred = model(X[cur_idx].to(device))
            loss += loss_fn(y_pred,y[cur_idx]).item()*cur_idx.shape[0]
            cur_re = torch.abs(y_pred-y[cur_idx])/torch.max(y[cur_idx],1)[0][:,None]
            for j in range(4):
                L1_re[j] += torch.sum(torch.mean(cur_re[:,j*n_resolution:(j+1)*n_resolution],1)).item()
            all_y_pred[cur_idx] = y_pred.detach().to("cpu").numpy()
    loss = loss/X.shape[0]
    L1_re = L1_re/X.shape[0]
    return loss, L1_re, all_y_pred

def run_experiment(task_name,seed_list=[seed],n_epoch=n_epoch,n_resolution=n_resolution,save_dir="./ML_result/"):
    result_name = ["train_loss",
                   "val_loss","val_S0","val_S1","val_S2","val_S3",
                   "test_loss","test_S0","test_S1","test_S2","test_S3"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print("Running %s"%task_name)
    for seed in seed_list:
        torch.manual_seed(seed)
        model = ResNet18(num_classes=n_resolution*4)
        model = model.to(device)
        loss_fn=MSELoss()
        optimizer=Adam(model.parameters(),lr=0.0001)
        result=np.zeros((n_epoch//10+1,len(result_name)),dtype=np.float32)
        
        first_stage=round(n_epoch*0.5)
        for i in range(first_stage):
            result[i//10,0]=train(model,x_train,y_train,optimizer,loss_fn)
            if i%10==0:
                result[i//10,1], result[i//10,2:6], y_val_pred = evaluate(model,x_val,y_val,loss_fn)
                result[i//10,6], result[i//10,7:], y_test_pred = evaluate(model,x_test,y_test,loss_fn)
                print(i,result[i//10])
                sys.stdout.flush()
        
        torch.save(model,save_dir+task_name+"_seed%d"%seed+".pt")
        np.savez_compressed(save_dir+task_name+"_seed%d"%seed+".npz",y_val_pred,y_test_pred)

        #optimizer.param_groups[0]['lr'] = 0.0005
        for i in range(first_stage,n_epoch):
            result[i//10,0]=train(model,x_train,y_train,optimizer,loss_fn)
            if i%10==0:
                result[i//10,1], result[i//10,2:6], y_val_pred = evaluate(model,x_val,y_val,loss_fn)
                result[i//10,6], result[i//10,7:], y_test_pred = evaluate(model,x_test,y_test,loss_fn)
                print(i,result[i//10])
                sys.stdout.flush()
                if np.min(result[:i//10,1])>result[i//10,1]:
                    torch.save(model,save_dir+task_name+"_seed%d"%seed+".pt")
                    np.savez_compressed(save_dir+task_name+"_seed%d"%seed+".npz",y_val_pred,y_test_pred)

        result=pd.DataFrame(result,columns=result_name)
        save_filename=save_dir+task_name+"_seed%d"%seed+".csv"
        result.to_csv(save_filename,index=False)

        model = torch.load(save_dir+task_name+"_seed%d"%seed+".pt").to(device)

print("loading dataset")
sys.stdout.flush()
fo = np.load("./processed.npz")
x_train = torch.from_numpy(fo["arr_0"])
y_train = torch.from_numpy(fo["arr_1"]).to(device)
x_val = torch.from_numpy(fo["arr_2"])
y_val = torch.from_numpy(fo["arr_3"]).to(device)
x_test = torch.from_numpy(fo["arr_4"])
y_test = torch.from_numpy(fo["arr_5"]).to(device)

print("Training")
run_experiment("aug")
print("All finished!")
