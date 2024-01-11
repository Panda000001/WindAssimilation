"""
2维NS
"""
import numpy as np
import torch.optim.lr_scheduler
from pinn_model import *
from read_data import DataPointsDataset, EquationPointsDataset
from torch.utils.data import DataLoader

import os
from learning_schdule import ChainedScheduler
import time
import sys


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == 'cpu':
    raise ValueError("Device CPU, not GPU.")

para_search = False # 注意多台设备/计算节点 同时搜索需要使用同一个sweep_id

# 训练集参数Wind
L = 60 # Ref60, IAG50
U = 8  # Ref8, IAG15
Rey = 4.8e7  # Ref4.8e7, IAG5.259467e7
# rou = 1 # 不用给
Nt,dt = 100, 1 # Nt=100, transfer Nt=20
LIDAR_X = -10/L
LIDAR_Y= 0/L
# 一用于生成方程点，二用于计算 mse_eqns, 
eq_dict = {'Rey': Rey,
           'mtd': 'uniform', 
           'x_range': np.array([-240, 0])/L, 
           'y_range': np.array([-60, 60])/L, 
           't_range_bc': np.array([0, 100*dt])/(L/U), # Net properties: lb,ub
          #  't_range': np.array([0, Nt*dt])/(L/U), 
           't_range': np.array([100, 100+Nt*dt])/(L/U), # Modify when transfer
           'N_eq': np.array([81, 41, Nt]) # Modify when transfer
          }

train_dict = {"Rey": Rey, 
              "work_path": 'Wind_Ref_Transfer100s_test',
              "C_LoS":1,  # 1 or 0, if LoS data available
              "M_LoS": 11, "batch_LoS_data": 11*2*Nt, # *100, transfer*20
              # "LoS_h5_paths": ['../Sensor.U8.XY.h5/LIDAR_Beam_Down_p11.h5',
              #                  '../Sensor.U8.XY.h5/LIDAR_Beam_Up_p11.h5'],
              "LoS_h5_paths": ['../Sensor.U8.XY.h5/LIDAR_Beam_Down_t100-199s_p11.h5', # Transfer
                               '../Sensor.U8.XY.h5/LIDAR_Beam_Up_t100-199s_p11.h5'], 
              "C_Uvec":1, # if Uvec data available
              "M_Uvec": 3, "batch_Uvec_data": (3)*Nt, # *100, transfer*20
              # "Uvec_h5_paths": ['../Sensor.U8.XY.h5/M_XY_Mid_p3_t100_dt1.h5'],
              "Uvec_h5_paths": ['../Sensor.U8.XY.h5/M_XY_Mid_p3_t100-199s_dt1.h5'], # Transfer
              "C_uv":1,  # if uv data available
              "M_uv":3, "batch_uv_data": (3)*Nt, # *100, transfer*20
              # "uv_h5_paths": ['../Sensor.U8.XY.h5/M_XY_Down_p3_t100_dt1.h5'],
              "uv_h5_paths": ['../Sensor.U8.XY.h5/M_XY_Down_p3_t100-199s_dt1.h5'], # Transfer
              "C_p":1,   # if p data available
              "M_p": 3, "batch_p_data": 3*Nt, # *100, transfer*20
              # "p_h5_paths": ['../Sensor.U8.XY.h5/M_XY_Up_p3_t100_dt1.h5'],
              "p_h5_paths": ['../Sensor.U8.XY.h5/M_XY_Up_p3_t100-199s_dt1.h5'], # Transfer
              "net_type": 'u_v_p_nu', # Net type: u_v_p, psi_p, u_v_p_nu, psi_p_nu
              "hidden_layers": 10, # uncount in/out 10
              "layer_neuros": 128, # 128
              "coef_LoS": 1,
              "coef_Uvec": 1,
              "coef_uv": 1,
              "coef_p": 1,
              "coef_eqns": 0.1,

              "batch_eq": 100, #1000
              "learning_rate":1e-4,
              "lr_decay": 'no_decay', # no_decay, cosin_warmup_restart
              # cosin_warmup_restart define
              # "T_0": 10, 
              # "T_mul": 2, 
              # "eta_min": 1e-12, 
              # "gamma": 1.0, 
              # "max_lr": 0.001, 
              # "warmup_steps": 2,

              "epochs": 1000, # 200000/100
              "auto_save_epoch": 200, # 1000/10

              "LBFGS": False,
              "epochs_LBFGS": 40, # 100/1000
              "LBFGS_inner": 20, # 20
              "lr_BFGS":0.001,
              # validation
              # "real_path": ['../Sensor.U8.XY.h5/LIDAR_Center_XY_Z90_49x25_t100_dt1.h5'],
              "real_path": ['../Sensor.U8.XY.h5/Center_XY_49x25_t100-199s_dt1.h5'], # Transfer
              "model_load_path": 'None.pth', # 续算输出设置
              "loss_load_path": 'None.loss.csv', # 续算输出设置
              "lr_load_path": 'None.lr.csv', # 续算输出设置

              "transfer_learning":True,
              "frozen": 0,
              "transfer_model_load_path":'../Wind_Ref_Assim_Coef/Wind_u_v_p_nu_CoefEq0.1_LoS1_Uvec1_uv1_p1/PINN_pth_99999.pt',

              # wandb
              "project_name": "Ref_U8_XY_Transfer", # Assimilation_LoS-Uvec-p_Net_Test # IAG_SMO
              }

if para_search:
    import wandb
    wandb.login(key="xxxxxxxxxxxxxxxxxxxxx")

    # 定义参数搜索配置
    sweep_config = {
        'method': 'grid',  # grid网格搜索, random随机搜索
        'name': 'parameter_search',  # 实验名称

        'metric': {'goal': 'minimize', 'name': 'loss_all_epoch'},  # 优化目标

        'parameters': {
            # 'batch_data': {'values': [2200,]},
            'batch_eq': {'values': [1000]}, # 1000, 4000, 8000
            'learning_rate': {'values': [1e-4]},
            'layers': {'values': [10]},  # 10/4
            'neuros': {'values': [128]},  # 64, 128, 200
            'net_type':{'values': ['u_v_p_nu']}, # 'u_v_p','u_v_p_nu','psi_p','psi_p_nu'
            # for AssiSearch
            # 'R_data':{'values': ['Uvec','uv','p']},
            'C_LoS':{'values': [1]},
            'C_Uvec':{'values': [1]},
            'C_uv':{'values': [1]},
            'C_p':{'values': [1]},

            # coefficient of loss components
            'coef_LoS':{'values': [1]},
            'coef_Uvec':{'values': [1]},
            'coef_uv':{'values': [1]},
            'coef_p':{'values': [1]},
            'coef_eqns':{'values': [0.1]}, 

            # transfer learning
            'frozen':{'values': [0, 2, 4, 6, 8]},
        }
    }
    # 创建并运行参数搜索（第一台设备/第一个计算节点）
    sweep_id = wandb.sweep(sweep=sweep_config, project=train_dict["project_name"])  # 替换为你的项目名称
    print("sweep_id is ", sweep_id)
    # sweep_id = '29mp1moi' # 多台设备/计算节点 同时搜索需要使用同一个sweep_id


def validation(pinn_net, real_path):
    valid_dataset = DataPointsDataset(real_path,data_type='uvp')
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_dataset.Ns, shuffle=False)
    # 初始化结果存储
    ReL2_u = 0
    ReL2_v = 0 
    ReL2_p = 0 
    for [i, (valid_batch)] in enumerate(valid_dataloader):
        (x_batch, y_batch, t_batch, u_batch, v_batch, p_batch) = [tensor.to(device) for tensor in valid_batch]
        u_pred_batch, v_pred_batch, p_pred_batch = pinn_net.pred_uvp(x_batch, y_batch, t_batch)
        ReL2_u += (torch.norm(u_batch - u_pred_batch) / torch.norm(u_batch)).item()
        ReL2_v += (torch.norm(v_batch - v_pred_batch) / torch.norm(v_batch)).item()
        ReL2_p += (torch.norm(p_batch - p_pred_batch) / torch.norm(p_batch)).item()
        val_bch_num = i
    print("validation batch number is", val_bch_num)
    return ReL2_u/val_bch_num, ReL2_v/val_bch_num, ReL2_p/val_bch_num

def watch_nn(NN,ep):
    if para_search:
        for name, param in NN.named_parameters():
            wandb.log({f"param/{name}": param.data.cpu().numpy()}, step=ep)
            if param.grad is not None:
                wandb.log({f"grad/{name}": param.grad.data.cpu().numpy()}, step=ep)
    else: 
        pass

def train(): # para_search, train_dict, eq_dict
    # 根据实际需要调整寻优参数和固定参数
    if para_search:
        print("project_name is ", train_dict["project_name"])
        # 初始化wandb
        run = wandb.init(reinit=True,project=train_dict["project_name"], entity="aiforscience")
        # 使用wandb.config来访问超参数
        # batch_data = run.config.batch_data
        batch_eq = run.config.batch_eq
        
        layers = run.config.layers
        neuros = run.config.neuros
        net_type = run.config.net_type
        learning_rate = run.config.learning_rate
        # for AssiSearch
        C_LoS = run.config.C_LoS
        # R_data = run.config.R_data
        # if R_data == 'Uvec':
        #     C_Uvec, C_uv, C_p = 1, 0, 0
        # elif R_data == 'uv':
        #     C_Uvec, C_uv, C_p = 0, 1, 0
        # elif R_data == 'p':
        #     C_Uvec, C_uv, C_p = 0, 0, 1
        # else:
        #     print('R_data is ', R_data)
        #     raise ValueError("Unexpected config R_data ", R_data)

        C_Uvec = run.config.C_Uvec
        C_uv = run.config.C_uv
        C_p = run.config.C_p

        coef_LoS = run.config.coef_LoS
        coef_Uvec = run.config.coef_Uvec
        coef_uv = run.config.coef_uv
        coef_p = run.config.coef_p
        coef_eqns = run.config.coef_eqns

        # seaech net_type,batchsize,lr,L,N
        # work_path = train_dict["work_path"] + '/Wind_seed_{}_BchEq{}_lr{}_L{}_N{}'.format(net_type,batch_eq,learning_rate,layers,neuros)
        # search data assimilation type
        if train_dict["transfer_learning"]:
            frozen = run.config.frozen
            work_path = train_dict["work_path"] + '/Wind_case0_{}_frozen{}_LoS{}_Uvec{}_uv{}_p{}'.format(net_type,frozen,C_LoS,C_Uvec,C_uv,C_p)
        else:
            work_path = train_dict["work_path"] + '/Wind_case0_{}_LoS{}_Uvec{}_uv{}_p{}'.format(net_type,C_LoS,C_Uvec,C_uv,C_p)
        def wandb_loss_log(ep):
            wandb.log({ "loss_all_epoch": loss_all_epoch, 
                        "loss_data_LoS_epoch": loss_data_LoS_epoch, 
                        "loss_data_Uvec_epoch": loss_data_Uvec_epoch, 
                        "loss_data_uv_epoch": loss_data_uv_epoch, 
                        "loss_data_p_epoch": loss_data_p_epoch, 
                        "loss_eqns_epoch": loss_eqns_epoch}, step=ep)
        def wandb_valid_log(ep):
            print("Validation: ReL2_u ", valid_u,
                            ", ReL2_v ", valid_v, 
                            ", ReL2_p ", valid_p)
            wandb.log({ "ReL2_u": valid_u,
                        "ReL2_v": valid_v,
                        "ReL2_p": valid_p}, step=ep)
    else:
        batch_eq = train_dict["batch_eq"]
        layers = train_dict["hidden_layers"]
        neuros = train_dict["layer_neuros"]
        net_type = train_dict["net_type"]
        learning_rate = train_dict["learning_rate"]
        # work_path = train_dict["work_path"]+ '/Wind_{}_BchEq{}_lr{}_L{}_N{}'.format(net_type,batch_eq,learning_rate,layers,neuros)
        
        coef_LoS = train_dict["coef_LoS"]
        coef_Uvec = train_dict["coef_Uvec"]
        coef_uv = train_dict["coef_uv"]
        coef_p = train_dict["coef_p"]
        coef_eqns = train_dict["coef_eqns"]
        
        # for AssiSearch
        C_LoS = train_dict["C_LoS"]
        C_Uvec = train_dict["C_Uvec"]
        C_uv = train_dict["C_uv"]
        C_p = train_dict["C_p"]
        if train_dict["transfer_learning"]:
            frozen = train_dict["frozen"]
            work_path = train_dict["work_path"]+ '/Windtest_{}_frozen{}_BchEq{}_lr{}_L{}_N{}'.format(net_type,frozen,batch_eq,learning_rate,layers,neuros)
        else:
            work_path = train_dict["work_path"]+ '/Wind_{}_BchEq{}_lr{}_L{}_N{}'.format(net_type,batch_eq,learning_rate,layers,neuros)

        def wandb_loss_log(ep):
            pass
        def wandb_valid_log(ep):
            print("Validation: ReL2_u ", valid_u,
                            ", ReL2_v ", valid_v,
                            ", ReL2_p ", valid_p)

        

    batch_LoS_data = train_dict["batch_LoS_data"]
    batch_Uvec_data = train_dict["batch_Uvec_data"]
    batch_uv_data = train_dict["batch_uv_data"]
    batch_p_data = train_dict["batch_p_data"]

    epochs = train_dict["epochs"]

    LBFGS = train_dict["LBFGS"]
    epochs_LBFGS = train_dict["epochs_LBFGS"]
    LBFGS_inner = train_dict["LBFGS_inner"]
    lr_LBFGS = train_dict["lr_BFGS"]

    Rey = train_dict["Rey"]

    LoS_h5_paths = train_dict["LoS_h5_paths"]
    Uvec_h5_paths = train_dict["Uvec_h5_paths"]
    uv_h5_paths = train_dict["uv_h5_paths"]
    p_h5_paths = train_dict["p_h5_paths"]

    real_path = train_dict["real_path"]

    lb = np.array([eq_dict['x_range'][0], eq_dict['y_range'][0], eq_dict['t_range_bc'][0]])
    ub = np.array([eq_dict['x_range'][1], eq_dict['y_range'][1], eq_dict['t_range_bc'][1]])
    print("low_bound is ", lb, "up_bound is ", ub)

    if os.path.exists(train_dict["model_load_path"]):
        # pinn_net.load_state_dict(torch.load(train_dict["model_load_path"], map_location=device))
        pinn_net = torch.load(train_dict["model_load_path"])
        print("model: {} load done.".format(train_dict["model_load_path"]))
            
    if os.path.exists(train_dict["loss_load_path"]):
        losses = np.loadtxt(train_dict["loss_load_path"],delimiter=",").tolist()
        print("losses: {} load done.".format(train_dict["loss_load_path"]))
    else:
        losses = []

    if os.path.exists(train_dict["lr_load_path"]):
        lr = np.loadtxt(train_dict["lr_load_path"],delimiter=",").tolist()
        print("lr: {} load done.".format(train_dict["lr_load_path"]))
    else:
        lr = []

    if os.path.exists(work_path) == False:
            os.makedirs(work_path)

    filename_loss = work_path + '/loss.csv'
    filename_lr = work_path + '/lr.csv'

    # for AssiSearch 创建一个迭代器列表，如果数据存在，则添加对应的迭代器
    dataloader_list = []
    if C_LoS==1:
        LoS_dataset = DataPointsDataset(LoS_h5_paths,'LoS')
        LoS_dataloader = DataLoader(LoS_dataset, batch_size=batch_LoS_data, shuffle=True)
        dataloader_list.append(LoS_dataloader)
        print("LoS data batches: ", len(LoS_dataloader))
        if (len(LoS_dataloader) != 1):
            raise ValueError("LoS_dataloader do not have the expected length of 1.")
            
    if C_Uvec==1:
        Uvec_dataset = DataPointsDataset(Uvec_h5_paths,'Uvec')
        Uvec_dataloader = DataLoader(Uvec_dataset, batch_size=batch_Uvec_data, shuffle=True)
        dataloader_list.append(Uvec_dataloader)
        print("Uvec data batches: ", len(Uvec_dataloader))
        if (len(Uvec_dataloader) != 1):
            raise ValueError("Uvec_dataloader do not have the expected length of 1.")
    if C_uv==1:
        uv_dataset = DataPointsDataset(uv_h5_paths,'uv')
        uv_dataloader = DataLoader(uv_dataset, batch_size=batch_uv_data, shuffle=True)
        dataloader_list.append(uv_dataloader)
        print("uv data batches: ", len(uv_dataloader))
        if (len(uv_dataloader) != 1):
            raise ValueError("uv_dataloader do not have the expected length of 1.")
    if C_p==1:
        p_dataset = DataPointsDataset(p_h5_paths,'p')
        p_dataloader = DataLoader(p_dataset, batch_size=batch_p_data, shuffle=True)
        dataloader_list.append(p_dataloader)
        print("p data batches: ", len(p_dataloader))
        if (len(p_dataloader) != 1):
            raise ValueError("p_dataloader do not have the expected length of 1.")
    
    equation_dataset = EquationPointsDataset(eq_dict)

    # for SimpleSearch
    # LoS_dataset = DataPointsDataset(LoS_h5_paths,'LoS')
    # Uvec_dataset = DataPointsDataset(Uvec_h5_paths,'Uvec')
    # uv_dataset = DataPointsDataset(uv_h5_paths,'uv')
    # p_dataset = DataPointsDataset(p_h5_paths,'p')

    # LoS_dataloader = DataLoader(LoS_dataset, batch_size=batch_LoS_data, shuffle=True)
    # Uvec_dataloader = DataLoader(Uvec_dataset, batch_size=batch_Uvec_data, shuffle=True)
    # uv_dataloader = DataLoader(uv_dataset, batch_size=batch_uv_data, shuffle=True)
    # p_dataloader = DataLoader(p_dataset, batch_size=batch_p_data, shuffle=True)
    
    # print("LoS data batches: ", len(LoS_dataloader),
    #       "\nUvec data batches: ", len(Uvec_dataloader),
    #       "\nuv data batches: ", len(uv_dataloader),
    #       "\np data batches: ", len(p_dataloader),
    #       )
    # 三者的batches需要一致，因为后面要zip
    # if (len(LoS_dataloader) != 1) or (len(Uvec_dataloader) != 1) or (len(uv_dataloader) != 1) or (len(p_dataloader) != 1):
    #     raise ValueError("One or more DataLoaders do not have the expected length of 1.")
    # Net type: u_v_p, psi_p, u_v_p_nu, psi_p_nu
    if net_type == 'psi_p':
        layer_mat = [3] + layers * [neuros] + [2]
        print("'psi_p' layer_mat is ", layer_mat)
    elif net_type == 'u_v_p':
        layer_mat = [3] + layers * [neuros] + [3]
        print("'u_v_p' layer_mat is ", layer_mat)
    elif net_type == 'u_v_p_nu':
        layer_mat = [3] + layers * [neuros] + [4]
        print("'u_v_p_nu' layer_mat is ", layer_mat)
    elif net_type == 'psi_p_nu':
        layer_mat = [3] + layers * [neuros] + [3]
        print("'psi_p_nu' layer_mat is ", layer_mat)
    else:
        raise ValueError("Invalid net_type, should be u_v_p, psi_p, u_v_p_nu, psi_p_nu")

    # Transfer learning: load model
    if train_dict["transfer_learning"]:
        transfer_model_load_path = train_dict["transfer_model_load_path"]
        pinn_net = PINN_Net(layer_mat, net_type, lb, ub, device)  # 使用与保存模型时相同的参数初始化模型
        pinn_net.load_state_dict(torch.load(transfer_model_load_path))  # 加载模型参数
        pinn_net.to(device)  # 将模型移至GPU，如果使用CPU则改为 'cpu'

        # frozen = 3
        for i, layer in enumerate(pinn_net.base):
            if i // 2 < frozen:  # 每个线性层和激活层被视为单独的层
                for param in layer.parameters():
                    param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pinn_net.parameters()), lr=learning_rate)
    # ordinary
    else:
        pinn_net = PINN_Net(layer_mat, net_type, lb, ub, device)
        print("Total parameter: ", count_params(pinn_net))
        pinn_net = pinn_net.to(device)
        optimizer = torch.optim.Adam(pinn_net.parameters(), lr=learning_rate)

    if train_dict["lr_decay"] == 'no_decay':
        save_epoch_list = np.arange(train_dict["auto_save_epoch"]-1,train_dict["epochs"],train_dict["auto_save_epoch"])
        def scheduler_step():
            pass
        print("lr_decay is ", train_dict["lr_decay"])
        
    elif train_dict["lr_decay"] == 'cosin_warmup_restart':
        save_epoch_list = []
        save_ep = 0
        i = 0
        ti = train_dict["T_0"]
        while save_ep <= train_dict["epochs"]:
            save_ep = save_ep + ti
            save_epoch_list.append(save_ep-1)
            i += 1
            ti = ti * train_dict["T_mul"]
            
        save_epoch_list = np.array(save_epoch_list)

        scheduler = ChainedScheduler( optimizer, 
                                      T_0 = train_dict["T_0"], 
                                      T_mul = train_dict["T_mul"], 
                                      eta_min = train_dict["eta_min"], 
                                      gamma = train_dict["gamma"], 
                                      max_lr = train_dict["max_lr"], 
                                      warmup_steps = train_dict["warmup_steps"])
        def scheduler_step():
            scheduler.step()

        print("lr_decay is ", train_dict["lr_decay"])
    else:
        raise ValueError("Unsupported lr_decay, please give 'no_decay' or 'cosin_warmup_restart'")
    
    epoch_start_time = time.time()
    for epoch in range(epochs):
        loss_all_epoch = 0
        loss_data_LoS_epoch = 0
        loss_data_Uvec_epoch = 0
        loss_data_uv_epoch = 0
        loss_data_p_epoch = 0
        loss_eqns_epoch = 0
        for [i_data, data_batch] in enumerate(zip(*dataloader_list)):
            step_start_time = time.time()
            id_dataloader = 0
            # LoS data
            # if C_LoS==1:
            LoS_batch = data_batch[id_dataloader]
            (x_LoS_data_bch, y_LoS_data_bch, t_LoS_data_bch, u_LoS_data_bch) = [
                tensor.requires_grad_(True).to(device) for tensor in LoS_batch]
            mse_LoS_predict = pinn_net.data_mse_LoS(x_LoS_data_bch, y_LoS_data_bch, t_LoS_data_bch, 
                                                    u_LoS_data_bch, LIDAR_X, LIDAR_Y)
            loss_data_LoS_epoch += mse_LoS_predict.item()
            id_dataloader += 1
            # else:
            #     mse_LoS_predict = 0
            # Uvec data
            # if C_Uvec==1:
            Uvec_batch = data_batch[id_dataloader]
            (x_Uvec_data_bch, y_Uvec_data_bch, t_Uvec_data_bch, u_mag_data_bch, u_dir_data_bch) = [
                tensor.requires_grad_(True).to(device) for tensor in Uvec_batch]
            mse_Uvec_predict = pinn_net.data_mse_vec(x_Uvec_data_bch, y_Uvec_data_bch, t_Uvec_data_bch,
                                                    u_mag_data_bch, u_dir_data_bch)
            loss_data_Uvec_epoch += mse_Uvec_predict.item()
            id_dataloader += 1
            # else:
            #     mse_Uvec_predict = 0
            # uv data
            # if C_uv==1:
            uv_batch = data_batch[id_dataloader]
            (x_uv_data_bch, y_uv_data_bch, t_uv_data_bch, u_uv_data_bch, v_uv_data_bch) = [
                tensor.requires_grad_(True).to(device) for tensor in uv_batch]
            mse_uv_predict = pinn_net.data_mse_uv(x_uv_data_bch, y_uv_data_bch, t_uv_data_bch,
                                                  u_uv_data_bch, v_uv_data_bch)
            loss_data_uv_epoch += mse_uv_predict.item()
            id_dataloader += 1
            # else:
            #     mse_uv_predict = 0
            # p data
            # if C_p==1:
            p_batch = data_batch[id_dataloader]
            (x_p_data_bch, y_p_data_bch, t_p_data_bch, p_p_data_bch) = [
                tensor.requires_grad_(True).to(device) for tensor in p_batch]
            mse_p_predict = pinn_net.data_mse_p(x_p_data_bch, y_p_data_bch, t_p_data_bch, p_p_data_bch)
            loss_data_p_epoch += mse_p_predict.item()
            # else:
            #     mse_p_predict = 0
            
        # for [i_data, (LoS_batch,
        #               Uvec_batch,
        #               uv_batch,
        #               p_batch)] in enumerate(zip(LoS_dataloader,
        #                                          Uvec_dataloader,
        #                                          uv_dataloader,
        #                                          p_dataloader)):
            # (x_LoS_data_bch, 
            #  y_LoS_data_bch, 
            #  t_LoS_data_bch, 
            #  u_LoS_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in LoS_batch]
            
            # (x_Uvec_data_bch, 
            #  y_Uvec_data_bch, 
            #  t_Uvec_data_bch, 
            #  u_mag_data_bch,
            #  u_dir_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in Uvec_batch]
            
            # (x_uv_data_bch, 
            #  y_uv_data_bch, 
            #  t_uv_data_bch, 
            #  u_uv_data_bch,
            #  v_uv_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in uv_batch]
            
            # (x_p_data_bch, 
            #  y_p_data_bch, 
            #  t_p_data_bch, 
            #  p_p_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in p_batch]
            
            # 如果遍历方程点，则启动这里的for，把下一行直至end for前面的内容都放进循环
            # for [i_eqns, (eqns_batch)] in enumerate(equation_dataloader): 
                # eqns_batch = [tensor.requires_grad_(True).to(device) for tensor in eqns_batch]
            # 如果不遍历方程点start
            i_eqns = 0
            random_indices = torch.randperm(len(equation_dataset))[:batch_eq]
            random_data = [equation_dataset[i] for i in random_indices]
            x_eqns_bch, y_eqns_bch, t_eqns_bch = [torch.stack(tensors).requires_grad_(True).to(device) for tensors in zip(*random_data)]
            # 如果不遍历方程点end

            # 清除梯度
            optimizer.zero_grad()

            # mse_LoS_predict = pinn_net.data_mse_LoS(x_LoS_data_bch, y_LoS_data_bch, t_LoS_data_bch,
            #                                         u_LoS_data_bch, LIDAR_X, LIDAR_Y)
            # mse_Uvec_predict = pinn_net.data_mse_vec(x_Uvec_data_bch, y_Uvec_data_bch, t_Uvec_data_bch, 
            #                                          u_mag_data_bch, u_dir_data_bch)
            # mse_uv_predict = pinn_net.data_mse_uv(x_uv_data_bch, y_uv_data_bch, t_uv_data_bch, 
            #                                       u_uv_data_bch, v_uv_data_bch)
            # mse_p_predict = pinn_net.data_mse_p(x_p_data_bch, y_p_data_bch, t_p_data_bch, p_p_data_bch)
            mse_equation = pinn_net.eqns_mse(x_eqns_bch, y_eqns_bch, t_eqns_bch, **eq_dict)
            loss = coef_LoS * mse_LoS_predict + \
                   coef_Uvec * mse_Uvec_predict + \
                   coef_uv * mse_uv_predict + \
                   coef_p * mse_p_predict + \
                   coef_eqns * mse_equation

            loss.backward()
            optimizer.step()
            step_end_time = time.time()
            # 使用.item()来获取标量值
            loss_all = loss.item()
            # loss_data_LoS = mse_LoS_predict.item()
            # loss_data_Uvec = mse_Uvec_predict.item()
            # loss_data_uv = mse_uv_predict.item()
            # loss_data_p = mse_p_predict.item()
            loss_eqns = mse_equation.item()

#             print("Epoch:", epoch + 1, 
#                   "Data_bch:", i_data + 1,
# #                  "Eqns_bch:", i_eqns + 1,
#                   "Training Loss:", loss_all,
#                   # "Data_LoS Loss:", loss_data_LoS,
#                   # "Data_Uvec Loss:", loss_data_Uvec,
#                   # "Data_p Loss:", loss_data_p,
# #                  "Eqns Loss:", loss_eqns,
#                   "Step time", step_end_time-step_start_time)
#             sys.stdout.flush()  # 刷新输出

            loss_all_epoch += loss_all
            # loss_data_LoS_epoch += loss_data_LoS
            # loss_data_Uvec_epoch += loss_data_Uvec
            # loss_data_uv_epoch += loss_data_uv
            # loss_data_p_epoch += loss_data_p
            loss_eqns_epoch += loss_eqns
            # 如果遍历方程点，则end for here

        losses.append([loss_all_epoch, loss_data_LoS_epoch, loss_data_Uvec_epoch, 
                       loss_data_uv_epoch, loss_data_p_epoch, 
                       loss_eqns_epoch
                       ])
        scheduler_step()
        # lr_now = scheduler.get_lr()[0]
        lr_now = optimizer.param_groups[0]['lr']
        lr.append(lr_now)
        # print("lr_now is ", lr_now)
        
        # 在训练循环中，使用wandb.log来记录指标
        wandb_loss_log(epoch)
        
        if epoch in save_epoch_list:
            epoch_end_time = time.time()
            # parameter and gradient
            watch_nn(pinn_net, epoch)
            
            print("Epoch:", epoch + 1, 
                  "Data_bch:", i_data + 1,
                  "Eqns_bch:", i_eqns + 1,
                  "Training Loss:", loss_all,
                  "Data_LoS Loss:", loss_data_LoS_epoch,
                  "Data_Uvec Loss:", loss_data_Uvec_epoch,
                  "Data_uv Loss:", loss_data_uv_epoch,
                  "Data_p Loss:", loss_data_p_epoch,
                  "Eqns Loss:", loss_eqns_epoch,
                  "Time cost ", epoch_end_time-epoch_start_time)
            sys.stdout.flush()  # 刷新输出
            
            valid_u, valid_v, valid_p = validation(pinn_net, real_path)
            wandb_valid_log(epoch)
            torch.save(pinn_net, work_path +'/PINN_pth_{}.pth'.format(epoch))
            torch.save(pinn_net.state_dict(), work_path +'/PINN_pth_{}.pt'.format(epoch))
            print("Model in epoch {} save to {}".format(epoch, work_path))
            epoch_start_time = time.time() # update start_time
        else:
            continue
    print("Adam oK")

    if LBFGS:
        # 使用L-BFGS优化器
        min_lbfgs = 1.0 * np.finfo(float).eps
        optimizer = torch.optim.LBFGS(pinn_net.parameters(), 
                                      lr = lr_LBFGS, 
                                      max_iter = LBFGS_inner, 
                                      max_eval = LBFGS_inner, 
                                      tolerance_grad = min_lbfgs, 
                                      tolerance_change = min_lbfgs, 
                                      history_size = 50, 
                                      line_search_fn=None)  
    
        for epoch in range(epochs_LBFGS):
            loss_all_epoch = 0
            loss_data_LoS_epoch = 0
            loss_data_Uvec_epoch = 0
            loss_data_uv_epoch = 0
            loss_data_p_epoch = 0
            loss_eqns_epoch = 0
            for [i_data, data_batch] in enumerate(zip(*dataloader_list)):
                id_dataloader = 0
                # LoS data
                if C_LoS==1:
                    LoS_batch = data_batch[id_dataloader]
                    (x_LoS_data_bch, y_LoS_data_bch, t_LoS_data_bch, u_LoS_data_bch) = [
                        tensor.requires_grad_(True).to(device) for tensor in LoS_batch]
                    mse_LoS_predict = pinn_net.data_mse_LoS(x_LoS_data_bch, y_LoS_data_bch, t_LoS_data_bch, 
                                                            u_LoS_data_bch, LIDAR_X, LIDAR_Y)
                    loss_data_LoS_epoch += mse_LoS_predict.item()
                    id_dataloader += 1
                else:
                    mse_LoS_predict = 0
                # Uvec data
                if C_Uvec==1:
                    Uvec_batch = data_batch[id_dataloader]
                    (x_Uvec_data_bch, y_Uvec_data_bch, t_Uvec_data_bch, u_mag_data_bch, u_dir_data_bch) = [
                        tensor.requires_grad_(True).to(device) for tensor in Uvec_batch]
                    mse_Uvec_predict = pinn_net.data_mse_vec(x_Uvec_data_bch, y_Uvec_data_bch, t_Uvec_data_bch,
                                                            u_mag_data_bch, u_dir_data_bch)
                    loss_data_Uvec_epoch += mse_Uvec_predict.item()
                    id_dataloader += 1
                else:
                    mse_Uvec_predict = 0
                # uv data
                if C_uv==1:
                    uv_batch = data_batch[id_dataloader]
                    (x_uv_data_bch, y_uv_data_bch, t_uv_data_bch, u_uv_data_bch, v_uv_data_bch) = [
                        tensor.requires_grad_(True).to(device) for tensor in uv_batch]
                    mse_uv_predict = pinn_net.data_mse_uv(x_uv_data_bch, y_uv_data_bch, t_uv_data_bch,
                                                          u_uv_data_bch, v_uv_data_bch)
                    loss_data_uv_epoch += mse_uv_predict.item()
                    id_dataloader += 1
                else:
                    mse_uv_predict = 0
                # p data
                if C_p==1:
                    p_batch = data_batch[id_dataloader]
                    (x_p_data_bch, y_p_data_bch, t_p_data_bch, p_p_data_bch) = [
                        tensor.requires_grad_(True).to(device) for tensor in p_batch]
                    mse_p_predict = pinn_net.data_mse_p(x_p_data_bch, y_p_data_bch, t_p_data_bch, p_p_data_bch)
                    loss_data_p_epoch += mse_p_predict.item()
                else:
                    mse_p_predict = 0
            
            # for [i_data, (LoS_batch,
            #               Uvec_batch,
            #               uv_batch,
            #               p_batch)] in enumerate(zip(LoS_dataloader,
            #                                          Uvec_dataloader,
            #                                          uv_dataloader,
            #                                          p_dataloader)):
            #     (x_LoS_data_bch, 
            #      y_LoS_data_bch, 
            #      t_LoS_data_bch, 
            #      u_LoS_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in LoS_batch]

            #     (x_Uvec_data_bch, 
            #      y_Uvec_data_bch, 
            #      t_Uvec_data_bch, 
            #      u_mag_data_bch,
            #      u_dir_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in Uvec_batch]
                
            #     (x_uv_data_bch, 
            #      y_uv_data_bch, 
            #      t_uv_data_bch, 
            #      u_uv_data_bch,
            #      v_uv_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in uv_batch]

            #     (x_p_data_bch, 
            #      y_p_data_bch, 
            #      t_p_data_bch, 
            #      p_p_data_bch) = [tensor.requires_grad_(True).to(device) for tensor in p_batch]

                # 如果遍历方程点，则启动这里的for，把下一行直至end for前面的内容都放进循环
                # for [i_eqns, (eqns_batch)] in enumerate(equation_dataloader):
                #     eqns_batch = [tensor.requires_grad_(True).to(device) for tensor in eqns_batch]
                # 如果不遍历方程点start
                i_eqns = 0
                random_indices = torch.randperm(len(equation_dataset))[:batch_eq]
                random_data = [equation_dataset[i] for i in random_indices]
                x_eqns_bch, y_eqns_bch, t_eqns_bch = [torch.stack(tensors).requires_grad_(True).to(device) for tensors in zip(*random_data)]
                # 如果不遍历方程点end
                
                # 定义一个字典来存储损失值
                loss_values = {"loss_all": 0, "loss_data_LoS": 0, "loss_data_uv":0, "loss_data_p":0, "loss_eqns": 0, "LBFGS_No": 0}
                def closure():
                    # 清除梯度
                    optimizer.zero_grad()
                    if C_LoS==1:
                        mse_LoS_predict = pinn_net.data_mse_LoS(x_LoS_data_bch, y_LoS_data_bch, t_LoS_data_bch,
                                                                u_LoS_data_bch, LIDAR_X, LIDAR_Y)
                        loss_values["loss_data_LoS"] = mse_LoS_predict.item()
                    else:
                        mse_LoS_predict = 0
                        loss_values["loss_data_LoS"] = 0
                    
                    if C_Uvec==1:
                        mse_Uvec_predict = pinn_net.data_mse_vec(x_Uvec_data_bch, y_Uvec_data_bch, t_Uvec_data_bch, 
                                                                 u_mag_data_bch, u_dir_data_bch)
                        loss_values["loss_data_Uvec"] = mse_Uvec_predict.item()
                    else:
                        mse_Uvec_predict = 0
                        loss_values["loss_data_Uvec"] = 0

                    if C_uv==1:
                        mse_uv_predict = pinn_net.data_mse_uv(x_uv_data_bch, y_uv_data_bch, t_uv_data_bch, 
                                                              u_uv_data_bch, v_uv_data_bch)
                        loss_values["loss_data_uv"] = mse_uv_predict.item()
                    else:
                        mse_uv_predict = 0
                        loss_values["loss_data_uv"] = 0
                    
                    if C_p==1:
                        mse_p_predict = pinn_net.data_mse_p(x_p_data_bch, y_p_data_bch, t_p_data_bch, p_p_data_bch)
                        loss_values["loss_data_p"] = mse_p_predict.item()
                    else:
                        mse_p_predict = 0
                        loss_values["loss_data_p"] = 0
                    mse_equation = pinn_net.eqns_mse(x_eqns_bch, y_eqns_bch, t_eqns_bch, **eq_dict)
                    loss = coef_LoS * mse_LoS_predict + \
                           coef_Uvec * mse_Uvec_predict + \
                           coef_uv * mse_uv_predict + \
                           coef_p * mse_p_predict + \
                           coef_eqns * mse_equation

                    loss.backward()
                    # 使用.item()来获取标量值
                    # 更新字典中的损失值
                    loss_values["LBFGS_No"] += 1
                    loss_values["loss_all"] = loss.item()
                    # loss_values["loss_data_LoS"] = mse_LoS_predict.item()
                    # loss_values["loss_data_Uvec"] = mse_Uvec_predict.item()
                    # loss_values["loss_data_uv"] = mse_uv_predict.item()
                    # loss_values["loss_data_p"] = mse_p_predict.item()
                    loss_values["loss_eqns"] = mse_equation.item()

                    return loss
                
                optimizer.step(closure)
                if (epoch + 1) % 10 == 0:
                    # 手动记录参数和梯度
                    watch_nn(pinn_net, epochs + epoch)

                    print("LBFGS Epoch:", (epoch + 1), 
                          "Data_bch:", i_data + 1,
                          "Eqns_bch:", i_eqns + 1, 
                          "LBFGS_No",loss_values["LBFGS_No"],
                          "Training Loss:", loss_values["loss_all"],
                          "Data_LoS Loss:", loss_values["loss_data_LoS"],
                          "Data_Uvec Loss:", loss_values["loss_data_Uvec"],
                          "Data_uv Loss:", loss_values["loss_data_uv"],
                          "Data_p Loss:", loss_values["loss_data_p"],
                          "Eqns Loss:", loss_values["loss_eqns"])
                    sys.stdout.flush()  # 刷新输出

                loss_all_epoch += loss_values["loss_all"]
                loss_data_LoS_epoch += loss_values["loss_data_LoS"]
                loss_data_Uvec_epoch += loss_values["loss_data_Uvec"]
                loss_data_uv_epoch += loss_values["loss_data_uv"]
                loss_data_p_epoch += loss_values["loss_data_p"]
                loss_eqns_epoch += loss_values["loss_eqns"]
                # 如果遍历方程点，则end for here

                losses.append([loss_all_epoch, loss_data_LoS_epoch, loss_data_Uvec_epoch, 
                               loss_data_uv_epoch, loss_data_p_epoch, loss_eqns_epoch])
                wandb_loss_log(epochs + epoch)

                lr_now = optimizer.param_groups[0]['lr']
                # print('lr_now is ',lr_now)
                lr.append(lr_now)

        valid_u, valid_v, valid_p = validation(pinn_net, real_path)
        wandb_valid_log(epochs + epoch)

    torch.save(pinn_net.state_dict(), work_path +'/PINN_pth_finally.pt')
    torch.save(pinn_net, work_path +'/PINN_pth_finally.pth')
    print("np.array(losses).shape",np.array(losses).shape)
    np.savetxt(filename_loss,np.array(losses),delimiter=",",header='loss_all, loss_data_LoS, loss_data_p, loss_equation')
    np.savetxt(filename_lr,np.array(lr),delimiter=",",header='lr')
    end_time = time.time()
    print('Time used : %fs' % (end_time-start_time))
    if para_search:
        run.finish()

if __name__ == "__main__":
    start_time = time.time()
    if para_search:
        print("!!!!sweep_id is ", sweep_id, "project_name is ", train_dict["project_name"])
        wandb.agent(sweep_id, function = train, project=train_dict["project_name"])
    else:
        train()
    # os.system("/usr/bin/shutdown") # 运行完自动关机

    
    