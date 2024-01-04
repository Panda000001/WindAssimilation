"""
@author: Chang Yan
"""
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from pyDOE import lhs

class DataPointsDataset(Dataset):
    def __init__(self, h5_paths, data_type, t_jump=None, t_start=None, t_end=None):
        """data_type is uv,p,LoS"""
        self.data_type = data_type
        self.x_data = []
        self.y_data = []
        self.t_data = []
        if self.data_type == 'uvp':
            self.u_data = []
            self.v_data = []
            self.p_data = []
        if self.data_type == 'uv':
            self.u_data = []
            self.v_data = []
        if self.data_type == 'p':
            self.p_data = []
        if self.data_type == 'LoS':
            self.u_Los_data = []
        if self.data_type == 'Uvec':
            self.u_mag_data = []
            self.u_dir_data = []

        for h5_path in h5_paths:
            with h5py.File(h5_path, 'r') as hf:
                self.Ns = np.array(hf['x_all']).shape[0]
                self.Nt = np.array(hf['t_all']).shape[0]

                self.x_data.append(np.tile(np.array(hf['x_all']), (1, self.Nt)).flatten()[:, None])
                self.y_data.append(np.tile(np.array(hf['y_all']), (1, self.Nt)).flatten()[:, None])
                self.t_data.append(np.tile(np.array(hf['t_all'][t_start:t_end:t_jump]), (self.Ns, 1)).flatten()[:, None])
                if self.data_type == 'uvp':
                    self.u_data.append(np.array(hf['u_all']).flatten()[:, None])
                    self.v_data.append(np.array(hf['v_all']).flatten()[:, None])
                    self.p_data.append(np.array(hf['p_all']).flatten()[:, None])
                if self.data_type == 'uv':
                    self.u_data.append(np.array(hf['u_all']).flatten()[:, None])
                    self.v_data.append(np.array(hf['v_all']).flatten()[:, None])
                if self.data_type == 'p':
                    self.p_data.append(np.array(hf['p_all']).flatten()[:, None])
                if self.data_type == 'LoS':
                    self.u_Los_data.append(np.array(hf['u_LoS_all']).flatten()[:, None])
                if self.data_type == 'Uvec':
                    self.u_mag_data.append(np.array(hf['u_mag_all']).flatten()[:, None])
                    self.u_dir_data.append(np.array(hf['u_dir_all']).flatten()[:, None])
        
        self.x_data = torch.tensor(np.concatenate(self.x_data, axis=0), dtype=torch.float32)
        self.y_data = torch.tensor(np.concatenate(self.y_data, axis=0), dtype=torch.float32)
        self.t_data = torch.tensor(np.concatenate(self.t_data, axis=0), dtype=torch.float32)
        if self.data_type == 'uvp':
            self.u_data = torch.tensor(np.concatenate(self.u_data, axis=0), dtype=torch.float32)
            self.v_data = torch.tensor(np.concatenate(self.v_data, axis=0), dtype=torch.float32)
            self.p_data = torch.tensor(np.concatenate(self.p_data, axis=0), dtype=torch.float32)
        if self.data_type == 'uv':
            self.u_data = torch.tensor(np.concatenate(self.u_data, axis=0), dtype=torch.float32)
            self.v_data = torch.tensor(np.concatenate(self.v_data, axis=0), dtype=torch.float32)
        if self.data_type == 'p':
            self.p_data = torch.tensor(np.concatenate(self.p_data, axis=0), dtype=torch.float32)
        if self.data_type == 'LoS':
            self.u_Los_data = torch.tensor(np.concatenate(self.u_Los_data, axis=0), dtype=torch.float32)
        if self.data_type == 'Uvec':
            self.u_mag_data = torch.tensor(np.concatenate(self.u_mag_data, axis=0), dtype=torch.float32)
            self.u_dir_data = torch.tensor(np.concatenate(self.u_dir_data, axis=0), dtype=torch.float32)

    def __len__(self):
        # return len(self.x_data) // self.batch_data
        return len(self.x_data)


    def __getitem__(self, idx):
        if self.data_type == 'uvp':
            return self.x_data[idx], self.y_data[idx], self.t_data[idx], self.u_data[idx], self.v_data[idx], self.p_data[idx]
        if self.data_type == 'uv':
            return self.x_data[idx], self.y_data[idx], self.t_data[idx], self.u_data[idx], self.v_data[idx]
        if self.data_type == 'p':
            return self.x_data[idx], self.y_data[idx], self.t_data[idx], self.p_data[idx]
        if self.data_type == 'LoS':
            return self.x_data[idx], self.y_data[idx], self.t_data[idx], self.u_Los_data[idx]
        if self.data_type == 'Uvec':
            return self.x_data[idx], self.y_data[idx], self.t_data[idx], self.u_mag_data[idx], self.u_dir_data[idx]


class EquationPointsDataset(Dataset):
    def __init__(self, eq_dict):
        self.eq_dict = eq_dict
        # self.batch_eq = batch_eq
        
        if self.eq_dict['mtd'] == 'uniform':
            x_data = np.linspace(eq_dict['x_range'][0], eq_dict['x_range'][1], eq_dict['N_eq'][0])
            y_data = np.linspace(eq_dict['y_range'][0], eq_dict['y_range'][1], eq_dict['N_eq'][1])
            t_data = np.linspace(eq_dict['t_range'][0], eq_dict['t_range'][1], eq_dict['N_eq'][2])
            print("\nx_data shape is ", len(x_data), 
                  "\ny_data shape is ", len(y_data), 
                  "\nt_data shape is ", len(t_data))

            Ns = x_data.shape[0] * y_data.shape[0]
            Nt = t_data.shape[0]

            self.t_data = torch.tensor(np.tile(t_data, (Ns, 1)).flatten()[:, None], dtype=torch.float32)

            self.x_data, self.y_data = np.meshgrid(x_data, y_data)
            self.x_data = self.x_data.flatten()[:, None]
            self.y_data = self.y_data.flatten()[:, None]

            self.x_data = torch.tensor(np.tile(self.x_data, (1, Nt)).flatten()[:, None], dtype=torch.float32)
            self.y_data = torch.tensor(np.tile(self.y_data, (1, Nt)).flatten()[:, None], dtype=torch.float32)
            print("\nself.x_data shape is ", self.x_data.shape, 
                  "\nself.y_data shape is ", self.y_data.shape, 
                  "\nself.t_data shape is ", self.t_data.shape)

        elif self.eq_dict['mtd'] == 'lhs':
            total_points = eq_dict['N_eq'][0]
            lhs_array = lhs(3, samples=total_points)
            self.x_data = eq_dict['x_range'][0] + (eq_dict['x_range'][1] - eq_dict['x_range'][0]) * lhs_array[:, 0]
            self.y_data = eq_dict['y_range'][0] + (eq_dict['y_range'][1] - eq_dict['y_range'][0]) * lhs_array[:, 1]
            self.t_data = eq_dict['t_range'][0] + (eq_dict['t_range'][1] - eq_dict['t_range'][0]) * lhs_array[:, 2]
        
            print("x_data shape is ", len(x_data), 
                  "y_data shape is ", len(y_data), 
                  "t_data shape is ", len(t_data))

            Ns = x_data.shape[0] * y_data.shape[0]
            Nt = t_data.shape[0]

            self.t_data = torch.tensor(np.tile(t_data, (Ns, 1)).flatten()[:, None], dtype=torch.float32)

            self.x_data, self.y_data = np.meshgrid(x_data, y_data)
            self.x_data = self.x_data.flatten()[:, None]
            self.y_data = self.y_data.flatten()[:, None]

            self.x_data = torch.tensor(np.tile(self.x_data, (1, Nt)).flatten()[:, None], dtype=torch.float32)
            self.y_data = torch.tensor(np.tile(self.y_data, (1, Nt)).flatten()[:, None], dtype=torch.float32)
            print("\nself.x_data shape is ", self.x_data.shape, 
                  "\nself.y_data shape is ", self.y_data.shape, 
                  "\nself.t_data shape is ", self.t_data.shape,
                  "\nself.x_data dtype is ", self.x_data.dtype, 
                  "\nself.y_data dtype is ", self.y_data.dtype, 
                  "\nself.t_data dtype is ", self.t_data.dtype)
        else:
            raise ValueError("Invalid 'mtd'")
        
    def __len__(self):
        # return len(self.x_data) // self.batch_eq
        return len(self.x_data)


    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.t_data[idx]
