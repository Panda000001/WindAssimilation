"""
@author: Chang Yan
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

device = torch.device("cuda")

real_path = './Sensor.U8.XY.h5/Center_XY_Z90_49x25_t100_dt1.h5'


work_path = './model_save'
ep = 100000
pred_write_path = work_path+'/Pred_49x25_ep{}k.h5'.format(int(ep/1000))
pinn_net = torch.load(work_path+'/PINN_pth_{}.pth'.format(ep-1))

pinn_net.cuda()


Nx,Ny = 49,25
N_100, T_100 = Nx*Ny, 100 
batch_pred = True # default batchsize is Nx*Ny

with h5py.File(real_path, 'r') as hf:
    Ns = np.array(hf['x_all']).shape[0]
    Nt = np.array(hf['t_all']).shape[0]
    print("Ns is ", Ns, "Nt is ", Nt)
    x_all = np.array(hf['x_all'])
    y_all = np.array(hf['y_all'])
    t_all = np.array(hf['t_all'])
    u_all = np.array(hf['u_all'])
    v_all = np.array(hf['v_all'])
    p_all = np.array(hf['p_all'])
    print("x_all shape is ", x_all.shape,
          "y_all shape is ", y_all.shape,
          "t_all shape is ", t_all.shape,
          "u_all shape is ", u_all.shape,
          "v_all shape is ", v_all.shape)
    
x_pred_100 = np.tile(x_all, (1, Nt)).flatten()[:, None]
y_pred_100 = np.tile(y_all, (1, Nt)).flatten()[:, None]
t_pred_100 = np.tile(t_all, (Ns, 1)).flatten()[:, None]
x_pred_100 = torch.tensor(x_pred_100, dtype=torch.float32, device=device, requires_grad=True)
y_pred_100 = torch.tensor(y_pred_100, dtype=torch.float32, device=device, requires_grad=True)
t_pred_100 = torch.tensor(t_pred_100, dtype=torch.float32, device=device, requires_grad=True)

if batch_pred:
    class PredictionDataset(Dataset):
        def __init__(self, x, y, t):
            self.x = x
            self.y = y
            self.t = t

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.t[idx]

    batch_size = Nx*Ny

    dataset = PredictionDataset(x_pred_100, y_pred_100, t_pred_100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    p_pred_all = []
    u_pred_all = []
    v_pred_all = []
    
    for x_batch, y_batch, t_batch in dataloader:
        u_pred_batch, v_pred_batch, p_pred_batch = pinn_net.pred_uvp(x_batch, y_batch, t_batch)
        p_pred_all.append(p_pred_batch)
        u_pred_all.append(u_pred_batch)
        v_pred_all.append(v_pred_batch)
        
    p_pred_100 = torch.cat(p_pred_all, dim=0)
    u_pred_100 = torch.cat(u_pred_all, dim=0)
    v_pred_100 = torch.cat(v_pred_all, dim=0)
        
p_pred = p_pred_100.reshape(N_100, T_100).detach().cpu().numpy()
u_pred = u_pred_100.reshape(N_100, T_100).detach().cpu().numpy()
v_pred = v_pred_100.reshape(N_100, T_100).detach().cpu().numpy()

# save as .h5
with h5py.File(pred_write_path, 'w') as hf:
    hf.create_dataset("x_pred", data=x_all)
    hf.create_dataset("y_pred", data=y_all)
    hf.create_dataset("t_pred", data=t_all)
    hf.create_dataset("u_pred", data=u_pred)
    hf.create_dataset("v_pred", data=v_pred)
    hf.create_dataset("p_pred", data=p_pred)

