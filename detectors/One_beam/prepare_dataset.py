import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

filename = "processed.npz"
n_resolution = 401
seed = 324

def stokes_from_alphabeta(alpha,beta):
    result = np.zeros((alpha.shape[0],3),dtype=np.float32)
    result[:,0] = 0.5*(np.cos(4*alpha-4*beta)+np.cos(4*alpha))
    result[:,1] = 0.5*(np.sin(4*alpha)-np.sin(4*alpha-4*beta))
    result[:,2] = -np.sin(4*alpha-2*beta)
    return result

np.random.seed(seed)
shuffled_idx = np.arange(5000)
np.random.shuffle(shuffled_idx)

fo = np.load("./random_parameters.npz")
wl_array = fo["arr_0"]
slit_array = fo["arr_1"]
alpha_array = fo["arr_2"].astype(np.float32)/180*np.pi
beta_array = fo["arr_3"].astype(np.float32)/180*np.pi

all_S = stokes_from_alphabeta(alpha_array,beta_array)
all_y = np.zeros((5000,n_resolution*4),dtype = np.float32)
all_x = np.zeros((5000,1,448,448),dtype=np.float32)

for i in tqdm(range(50)):
    cur_specs_raw = np.load("./spectrum/spec%d.npz"%i)["arr_0"].astype(np.float32)
    fo = np.load("./data/data%d.npz"%i)
    cur_imgs = fo["arr_0"]
    cur_exps = fo["arr_1"]
    for j in range(100):
        all_y[i*100+j,:n_resolution] = np.interp(np.linspace(400,800,n_resolution,dtype=np.float32),
                                                 np.linspace(400,800,cur_specs_raw.shape[1],dtype=np.float32),
                                                 cur_specs_raw[j])
        all_x[i*100+j,0] = zoom(cur_imgs[j]/cur_exps[j],(448/cur_imgs.shape[1],448/cur_imgs.shape[2])).astype(np.float32)
all_y = all_y/np.max(all_y)
for i in range(3):
    all_y[:,(i+1)*n_resolution:(i+2)*n_resolution] = all_y[:,:n_resolution]*np.expand_dims(all_S[:,i],axis=1)
all_x = all_x/np.max(all_x)

all_x = all_x[shuffled_idx]
all_y = all_y[shuffled_idx]
np.savez_compressed("processed.npz",
                    all_x[:4500],all_y[:4500],
                    all_x[4500:4750],all_y[4500:4750],
                    all_x[4750:],all_y[4750:])
print("Finished...")
