import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
from scipy.ndimage import zoom
from tqdm import tqdm

def get_resultx(white,all_spec):
    bounds = (0, np.inf)
    cur_result = lsq_linear(all_spec.T,white,bounds=bounds).x
    result = np.zeros((10,cur_result.shape[0]))
    result[0] = cur_result
    for i in range(result.shape[0]):
        result[i] = savgol_filter(cur_result,i*5+5,1)
    result[:,610:] = 0
    return result
    
print("Loading monochromatic data")
imgs = []
for i in tqdm(range(1,9)):
    fo = np.load("./data%d.npz"%i)
    cur_imgs = fo["arr_0"].astype(np.float32)
    cur_exp = fo["arr_1"]
    for j in range(cur_exp.shape[0]):
        imgs.append(zoom(cur_imgs[j]/cur_exp[j],
                         (448/cur_imgs[j].shape[0],448/cur_imgs[j].shape[1])).astype(np.float32))

fo = np.load("./spectrums.npz")
all_spec = fo["arr_0"]

def get_train_aug(name,white_i):
    white_spec = np.load(name+"_spec.npz")["arr_0"]
    fo = np.load(name+"_imgs.npz")
    white_mean = np.mean(np.mean(fo["arr_0"],-1),-1)/fo["arr_1"]
    
    x_train_aug = []
    y_train_aug = []
    for i in tqdm(range(len(white_i))):
        resultx = get_resultx(white_spec[white_i[i]],all_spec)
        for j in range(10):
            cur_y = np.zeros_like(all_spec[0])
            cur_x = np.zeros((1,448,448))
            for k in range(resultx.shape[1]):
                cur_y += all_spec[k]*resultx[j,k]
                cur_x += imgs[k]*resultx[j,k]
            cur_x = white_mean[white_i[i]]/np.mean(cur_x)*cur_x
            cur_y = savgol_filter(cur_y,30,1)
            x_train_aug.append(cur_x)
            y_train_aug.append(cur_y)
    x_train_aug = np.stack(x_train_aug).astype(np.float32)
    y_train_aug = np.stack(y_train_aug).astype(np.float32)
    
    return x_train_aug,y_train_aug

print("Loading white")
white_i = [2,3,4,5,6,7]
name1="white"
x_aug1,y_aug1 = get_train_aug(name1,white_i)
print("Loading dye1")
dye1_i = [2,3,4,5,6,7,8,9]
name2 = "dye1"
x_aug2,y_aug2 = get_train_aug(name2,dye1_i)
print("Loading dye2")
name3 = "dye2"
dye2_i = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
x_aug3,y_aug3 = get_train_aug(name3,dye2_i)

x_train_aug = np.concatenate([x_aug1,x_aug2,x_aug3])
y_train_aug = np.concatenate([y_aug1,y_aug2,y_aug3])
np.savez_compressed("train_aug_new.npz",x_train_aug,y_train_aug)

def get_testval_aug(names,white_is):
    x_val_aug = []
    x_test_aug = []
    y_val_aug = []
    y_test_aug = []
    for n in range(len(names)):
        name = names[n]
        white_i = white_is[n]
        white_spec = np.load(name+"_spec.npz")["arr_0"].astype(np.float32)
        fo = np.load(name+"_imgs.npz")
        cur_imgs = fo["arr_0"]
        cur_exp = fo["arr_1"]
        white_imgs = np.zeros((len(white_i),1,448,448),dtype=np.float32)
        idx = 0
        for i in white_i:
            white_imgs[idx,0] = zoom(cur_imgs[i]/cur_exp[i],(448/cur_imgs.shape[1],448/cur_imgs.shape[2])).astype(np.float32)
            idx += 1
        test_i = []
        val_i = []
        for i in range(len(white_i)):
            if i%2 == 0:
                test_i.append(i)
            else:
                val_i.append(i)
        
        x_val_aug.append(white_imgs[val_i])
        y_val_aug.append(white_spec[val_i])
        x_test_aug.append(white_imgs[test_i])
        y_test_aug.append(white_spec[test_i])
    x_val_aug = np.concatenate(x_val_aug)
    y_val_aug = np.concatenate(y_val_aug)
    x_test_aug = np.concatenate(x_test_aug)
    y_test_aug = np.concatenate(y_test_aug)
        
    return x_val_aug,y_val_aug,x_test_aug,y_test_aug

x_val_aug,y_val_aug,x_test_aug,y_test_aug = get_testval_aug(["white","dye1","dye2"],[white_i,dye1_i,dye2_i])
np.savez_compressed("./valtest_aug.npz",x_val_aug,y_val_aug,x_test_aug,y_test_aug)

