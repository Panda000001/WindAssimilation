"""
@author: Chang Yan
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from pyDOE import lhs

import operator
from functools import reduce

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
    
class PINN_Net(nn.Module):
    def __init__(self, layer_mat, net_type, low_bound, up_bound, device='cuda:0'):
        super(PINN_Net, self).__init__()
        """Net type: u_v_p, psi_p, u_v_p_nu, psi_p_nu
        """

        self.layer_num = len(layer_mat) - 1
        self.net_type = net_type
        self.lowbound = nn.Parameter(torch.from_numpy(low_bound.astype(np.float32)), requires_grad=False)
        self.upbound = nn.Parameter(torch.from_numpy(up_bound.astype(np.float32)), requires_grad=False)
        self.device = device
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module("linear{:02d}".format(i), nn.Linear(layer_mat[i], layer_mat[i + 1]))
            self.base.add_module("Act{:02d}".format(i), nn.Tanh())
        self.base.add_module("linear{:02d}".format(self.layer_num - 1),
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        self.Initial_param()

        # 根据net_type选择不同的函数 data_mse_NetType_DataType
        if self.net_type == 'u_v_p':
            if layer_mat[-1] != 3:
                raise ValueError("Unmatched net_type and layer_num")
            self.data_mse_uvp = self.data_mse_UVP_uvp
            self.data_mse_uv = self.data_mse_UVP_uv
            self.data_mse_p = self.data_mse_UVP_p
            self.data_mse_LoS = self.data_mse_UVP_LoS
            self.data_mse_vec = self.data_mse_UVP_vec
            self.eqns_mse = self.eqns_mse_UVP
            self.pred_uvp = self.pred_UVP_uvp
        elif self.net_type == 'psi_p':
            if layer_mat[-1] != 2:
                raise ValueError("Unmatched net_type and layer_num")
            self.data_mse_uvp = self.data_mse_PsiP_uvp
            self.data_mse_uv = self.data_mse_PsiP_uv
            self.data_mse_p = self.data_mse_PsiP_p
            self.data_mse_LoS = self.data_mse_PsiP_LoS
            self.data_mse_vec = self.data_mse_PsiP_vec
            self.eqns_mse = self.eqns_mse_PsiP
            self.pred_uvp = self.pred_PsiP_uvp
        elif self.net_type == 'u_v_p_nu':
            if layer_mat[-1] != 4:
                raise ValueError("Unmatched net_type and layer_num")
            self.data_mse_uvp = self.data_mse_UVP_uvp
            self.data_mse_uv = self.data_mse_UVP_uv
            self.data_mse_p = self.data_mse_UVP_p
            self.data_mse_LoS = self.data_mse_UVP_LoS
            self.data_mse_vec = self.data_mse_UVP_vec
            self.eqns_mse = self.eqns_mse_UVPNu
            self.pred_uvp = self.pred_UVP_uvp
        elif self.net_type == 'psi_p_nu':
            if layer_mat[-1] != 3:
                raise ValueError("Unmatched net_type and layer_num")
            self.data_mse_uvp = self.data_mse_PsiP_uvp
            self.data_mse_uv = self.data_mse_PsiP_uv
            self.data_mse_p = self.data_mse_PsiP_p
            self.data_mse_LoS = self.data_mse_PsiP_LoS
            self.data_mse_vec = self.data_mse_PsiP_vec
            self.eqns_mse = self.eqns_mse_PsiPNu
            self.pred_uvp = self.pred_PsiP_uvp
        else:
            raise ValueError("Invalid net_type")

    # 0-1 norm of input variable
    # 对输入变量进行0-1归一化
    def zero_one_norm(self, X):
        X_norm = (X-self.lowbound)/(self.upbound-self.lowbound)
        return X_norm

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        X_norm = self.zero_one_norm(X)
        predict = self.base(X_norm)
        return predict

    # initialize
    # 对参数进行初始化
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('linear.weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('linear.bias'):
                nn.init.zeros_(param)

    # derive loss for data, net_type is u_v_p or u_v_p_nu
    # 类内方法：求数据点的loss, 网络输出是 u_v_p 或 u_v_p_nu

    def data_mse_UVP_uvp(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict
    
    def data_mse_UVP_uv(self, x, y, t, u, v):
        predict_out = self.forward(x, y, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict
    
    def data_mse_UVP_p(self, x, y, t, p):
        predict_out = self.forward(x, y, t)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(p_predict, p)
        return mse_predict
    
    def data_mse_UVP_LoS(self, x, y, t, u_LoS_data, LIDAR_X, LIDAR_Y):
        predict_out = self.forward(x, y, t) # shape 3
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        theta = -torch.arctan((LIDAR_Y-y)/(LIDAR_X-x))
        u_LoS_pred = u_predict*torch.cos(theta)-v_predict*torch.sin(theta)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_LoS_pred, u_LoS_data)
        return mse_predict
    
    def data_mse_UVP_vec(self, x, y, t, u_mag, u_dir):
        predict_out = self.forward(x, y, t) # shape 3
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        u_mag_pred = torch.sqrt(u_predict**2 + v_predict**2)
        u_dir_pred = torch.atan2(v_predict, u_predict)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_mag_pred, u_mag) + mse(u_dir_pred, u_dir)
        return mse_predict

    # derive loss for data, net_type is psi_p
    # 类内方法：求数据点的loss, 网络输出是 psi_p
    def data_mse_PsiP_uvp(self, x, y, t, u, v, p):
        predict_out = self.forward(x, y, t) # shape 2
        psi = predict_out[:, 0].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        p_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict
    
    def data_mse_PsiP_uv(self, x, y, t, u, v):
        predict_out = self.forward(x, y, t) # shape 2
        psi = predict_out[:, 0].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict
    
    def data_mse_PsiP_p(self, x, y, t, p):
        predict_out = self.forward(x, y, t) # shape 2
        p_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(p_predict, p)
        return mse_predict
    
    def data_mse_PsiP_LoS(self, x, y, t, u_LoS_data, LIDAR_X, LIDAR_Y):
        predict_out = self.forward(x, y, t) # shape 2
        psi = predict_out[:, 0].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        theta = -torch.arctan((LIDAR_Y-y)/(LIDAR_X-x))
        u_LoS_pred = u_predict*torch.cos(theta)-v_predict*torch.sin(theta)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_LoS_pred, u_LoS_data)
        return mse_predict
    
    def data_mse_PsiP_vec(self, x, y, t, u_mag, u_dir):
        predict_out = self.forward(x, y, t) # shape 2
        psi = predict_out[:, 0].reshape(-1, 1)
        u_predict = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v_predict = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        u_mag_pred = torch.sqrt(u_predict**2 + v_predict**2)
        u_dir_pred = torch.atan2(v_predict, u_predict)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_mag_pred, u_mag) + mse(u_dir_pred, u_dir)
        return mse_predict

    # derive loss for equation
    # 类内方法：求方程点的loss, 网络输出是 u_v_p
    def eqns_mse_UVP(self, x, y, t, **eq_dict):
        Rey = eq_dict["Rey"]
        predict_out = self.forward(x, y, t)
        # 获得预测的输出u,v,w,p,k,epsilon
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Rey * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Rey * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)
        return mse_equation
    
    def eqns_mse_UVPNu(self, x, y, t, **eq_dict):
        Rey = eq_dict["Rey"]
        predict_out = self.forward(x, y, t)
        # 获得预测的输出u,v,w,p,Nu
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        Nu = predict_out[:, 3].reshape(-1, 1)

        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Rey * (1 + Nu) * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Rey * (1 + Nu) * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)
        return mse_equation

    def eqns_mse_PsiP(self, x, y, t, **eq_dict):
        Rey = eq_dict["Rey"]
        predict_out = self.forward(x, y, t) # shape 2
        psi = predict_out[:, 0].reshape(-1, 1)
        u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        p = predict_out[:, 1].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Rey * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Rey * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation
    
    def eqns_mse_PsiPNu(self, x, y, t, **eq_dict):
        Rey = eq_dict["Rey"]
        predict_out = self.forward(x, y, t) # shape 2
        psi = predict_out[:, 0].reshape(-1, 1)
        u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
        v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        p = predict_out[:, 1].reshape(-1, 1)
        Nu = predict_out[:, 2].reshape(-1, 1)
        
        # Rey = L*U/(Nu*1e-5) # Nu的数值尺度与uvp尽量接近 # Nu=Nu*1e-5
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Rey * (1 + Nu) * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Rey * (1 + Nu) * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation
    
    # Forward propagation, used in validation
    def pred_UVP_uvp(self, x, y, t):
        predict_out = self.forward(x, y, t)
        u_pred = predict_out[:, 0].reshape(-1, 1)
        v_pred = predict_out[:, 1].reshape(-1, 1)
        p_pred = predict_out[:, 2].reshape(-1, 1)
        return u_pred, v_pred, p_pred
    
    def pred_PsiP_uvp(self, x, y, t):
        x.requires_grad_(True)
        y.requires_grad_(True)
        pinn_pred = self.forward(x, y, t)
        psi_pred = pinn_pred[:, 0].reshape(-1, 1)
        p_pred = pinn_pred[:, 1].reshape(-1, 1)
        u_pred = torch.autograd.grad(psi_pred, y, grad_outputs=torch.ones_like(psi_pred), retain_graph=True)[0]
        v_pred = -torch.autograd.grad(psi_pred, x, grad_outputs=torch.ones_like(psi_pred), retain_graph=True)[0]
        return u_pred, v_pred, p_pred       