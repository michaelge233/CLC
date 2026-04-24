import os
import sys
import time
import numpy as np
import torch
from scipy.ndimage import zoom
from torch import nn
import pandas as pd
from torch.nn import MSELoss
from torch.optim import Adam

start_time = time.time()
seed = 324
nn_input_size = 448
save_dir="./ML_result/"

print("Using seed",seed)
print("Using torch", torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
sys.stdout.flush()

np.random.seed(seed)
torch.manual_seed(seed)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
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
    def __init__(self,img_channels=1,block=BasicBlock,num_classes = 964):
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
                i_end=X.shape[0]
            else:
                i_end=i+batchsize
                
            y_pred=model(X[i_start:i_end])
            loss = loss_fn(y_pred, y[i_start:i_end])
            epoch_loss += loss.item()*(i_end-i_start)
            
        epoch_loss=epoch_loss/X.shape[0]
    return epoch_loss

def run_cnn_exp(task_name,seed_list=[324],n_epoch=5001,save_dir=save_dir):
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


print("Loading dataset")
print("Loading monochromatic data")
sys.stdout.flush()
all_y = np.load("./spectrums.npz")["arr_0"].astype(np.float32)
imgs = np.zeros((all_y.shape[0],1,nn_input_size,nn_input_size),dtype=np.float32)
for i in range(1,9):
    fo = np.load("./data%d.npz"%i)
    cur_imgs = fo["arr_0"].astype(np.float32)
    cur_exp = fo["arr_1"]
    for j in range(cur_exp.shape[0]):
        imgs[(i-1)*100+j,0] = zoom(cur_imgs[j]/cur_exp[j],
                                (448/cur_imgs[j].shape[0],448/cur_imgs[j].shape[1])).astype(np.float32)
imgs = np.stack(imgs)
x_norm = np.mean(imgs)
y_norm = np.max(all_y)

print("Loading broadband data")
sys.stdout.flush()
test_i = np.arange(5,all_y.shape[0],10)
val_i = np.arange(9,all_y.shape[0],10)
train_i = []
for i in range(all_y.shape[0]):
    if i%10 !=5 and i%10!=9:
        train_i.append(i)
train_i = np.array(train_i)

x_train = imgs[train_i]
x_val = imgs[val_i]
x_test = imgs[test_i]
y_train = all_y[train_i]
y_val = all_y[val_i]
y_test = all_y[test_i]

fo = np.load("./train_aug_new.npz")
x_train = np.concatenate([x_train,fo["arr_0"]])
y_train = np.concatenate([y_train,fo["arr_1"]])

fo = np.load("./valtest_aug.npz")
x_val = np.concatenate([x_val,fo["arr_0"]])
y_val = np.concatenate([y_val,fo["arr_1"]])
x_test = np.concatenate([x_test,fo["arr_0"]])
y_test = np.concatenate([y_test,fo["arr_1"]])

print("Converting to Pytorch")
sys.stdout.flush()
x_train = x_train/x_norm
x_val = x_val/x_norm
x_test = x_test/x_norm
y_train = y_train/y_norm
y_val = y_val/y_norm
y_test = y_test/y_norm

x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
x_val = torch.from_numpy(x_val).to(device)
y_val = torch.from_numpy(y_val).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)
loading_time = time.time()
print("Dataset processing time (s):",loading_time-start_time)


print("Start training...")
run_cnn_exp("ResNet",seed_list=[seed])
end_time = time.time()
print("Traing time (s):",end_time-loading_time)
print("Total time (s):",end_time-start_time)
print("All finished...")
