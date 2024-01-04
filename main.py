"""
@author: Chang Yan
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

# For parallel search in multi device, the sweep_id should be given as same. 
para_search = False # wandb parameter search

# Non-dimensional reference
L = 60 
U = 8  
Rey = 4.8e7  

dt = 1
LIDAR_X = -10/L
LIDAR_Y= 0/L

# generate equatioin points and calculate mse_eqns 
eq_dict = {'Rey': Rey,
           'mtd': 'uniform', 
           'x_range': np.array([-240, 0])/L, 
           'y_range': np.array([-60, 60])/L, 
           't_range': np.array([0, 100*dt])/(L/U), 
           'N_eq': np.array([81, 41, 100])
          }

# hyper-parameter setting
train_dict = {"Rey": Rey, 
              "work_path": 'Wind_Ref_Assim',
              "C_LoS":1,  # 1 or 0, if LoS data available
              "M_LoS": 11, "batch_LoS_data": 11*2*100,
              "LoS_h5_paths": ['../LIDAR.U8.XY.h5/LIDAR_Beam_Down_p11.h5',
                               '../LIDAR.U8.XY.h5/LIDAR_Beam_Up_p11.h5'],
              "C_Uvec":0, # if Uvec data available
              "M_Uvec": 3, "batch_Uvec_data": (3)*100, # Ref(5),IAG(4+6)
              "Uvec_h5_paths": ['../LIDAR.U8.XY.h5/M_XY_Mid_p3_t100_dt1.h5'],
              "C_uv":0,  # if uv data available
              "M_uv":3, "batch_uv_data": (3)*100,
              "uv_h5_paths": ['../LIDAR.U8.XY.h5/M_XY_Down_p3_t100_dt1.h5'],
              "C_p":0,   # if p data available
              "M_p": 3, "batch_p_data": 3*100,
              "p_h5_paths": ['../LIDAR.U8.XY.h5/M_XY_Up_p3_t100_dt1.h5'],
              "net_type": 'u_v_p_nu', # Net type: u_v_p, psi_p, u_v_p_nu, psi_p_nu
              "hidden_layers": 10, # uncount in/out 10
              "layer_neuros": 128, # 128
              "coef_LoS": 1,
              "coef_Uvec": 1,
              "coef_uv": 1,
              "coef_p": 1,
              "coef_eqns": 1,

              "batch_eq": 1000,
              "learning_rate":1e-4,
              "lr_decay": 'no_decay', # no_decay, cosin_warmup_restart
              # cosin_warmup_restart define
              # "T_0": 10, 
              # "T_mul": 2, 
              # "eta_min": 1e-12, 
              # "gamma": 1.0, 
              # "max_lr": 0.001, 
              # "warmup_steps": 2,

              "epochs": 100000, # 200000/100
              "auto_save_epoch": 1000, # 1000/10

              "LBFGS": False,
              "epochs_LBFGS": 40, # 100/1000
              "LBFGS_inner": 20, # 20
              "lr_BFGS":0.001,
              # validation
              "real_path": ['../LIDAR.U8.XY.h5/LIDAR_Center_XY_Z90_49x25_t100_dt1.h5'],
              "model_load_path": 'None.pth', # Checkpoint Resumption
              "loss_load_path": 'None.loss.csv', # Checkpoint Resumption
              "lr_load_path": 'None.lr.csv', # Checkpoint Resumption

              # wandb
              "project_name": "Ref_U8_XY_Assimilation", 
              }

if para_search:
    import wandb
    wandb.login(key="xxxxxxxxxxxxxxxxxxxxx") # register wandb to get the key

    # wandb search setup
    sweep_config = {
        'method': 'grid',  # grid, random
        'name': 'parameter_search', 

        'metric': {'goal': 'minimize', 'name': 'loss_all_epoch'}, 

        'parameters': {
            # 'batch_data': {'values': [2200,]},
            'batch_eq': {'values': [1000]}, # 1000, 4000, 8000
            'learning_rate': {'values': [1e-4]},
            'layers': {'values': [10]},  # 10/4
            'neuros': {'values': [128]},  # 64, 128, 200
            'net_type':{'values': ['psi_p']}, # 'u_v_p','u_v_p_nu','psi_p','psi_p_nu'
            # for AssiSearch
            # 'R_data':{'values': ['Uvec','uv','p']},
            'C_LoS':{'values': [1]},
            'C_Uvec':{'values': [0,1]},
            'C_uv':{'values': [0,1]},
            'C_p':{'values': [0,1]},

            # coefficient of loss components
            'coef_LoS':{'values': [1]},
            'coef_Uvec':{'values': [1]},
            'coef_uv':{'values': [1]},
            'coef_p':{'values': [1]},
            'coef_eqns':{'values': [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}, 
        }
    }

    # wandb will give the first agent a sweep_id
    sweep_id = wandb.sweep(sweep=sweep_config, project=train_dict["project_name"])  # 替换为你的项目名称
    print("sweep_id is ", sweep_id)
    # sweep_id = '29mp1moi' # More sevices for one sweep should use the same sweep_id


def validation(pinn_net, real_path):
    valid_dataset = DataPointsDataset(real_path,data_type='uvp')
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_dataset.Ns, shuffle=False)
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
    # adjust this part according to your demand
    if para_search:
        print("project_name is ", train_dict["project_name"])
        # initialize wandb
        run = wandb.init(reinit=True,project=train_dict["project_name"], entity="aiforscience")
        # use wandb.config to assign hyper-parameters
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
        work_path = train_dict["work_path"] + '/Wind_{}_LoS{}_Uvec{}_uv{}_p{}'.format(net_type,C_LoS,C_Uvec,C_uv,C_p)
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
        work_path = train_dict["work_path"]+ '/Wind_{}_BchEq{}_lr{}_L{}_N{}'.format(net_type,batch_eq,learning_rate,layers,neuros)
        
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

    lb = np.array([eq_dict['x_range'][0], eq_dict['y_range'][0], eq_dict['t_range'][0]])
    ub = np.array([eq_dict['x_range'][1], eq_dict['y_range'][1], eq_dict['t_range'][1]])
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

    dataloader_list = []
    if C_LoS==1:
        LoS_dataset = DataPointsDataset(LoS_h5_paths,'LoS')
        LoS_dataloader = DataLoader(LoS_dataset, batch_size=batch_LoS_data, shuffle=True)
        dataloader_list.append(LoS_dataloader)
        if (len(LoS_dataloader) != 1):
            raise ValueError("LoS_dataloader do not have the expected length of 1.")
        print("LoS data batches: ", len(LoS_dataloader))
            
    if C_Uvec==1:
        Uvec_dataset = DataPointsDataset(Uvec_h5_paths,'Uvec')
        Uvec_dataloader = DataLoader(Uvec_dataset, batch_size=batch_Uvec_data, shuffle=True)
        dataloader_list.append(Uvec_dataloader)
        if (len(Uvec_dataloader) != 1):
            raise ValueError("Uvec_dataloader do not have the expected length of 1.")
        print("Uvec data batches: ", len(Uvec_dataloader))
    if C_uv==1:
        uv_dataset = DataPointsDataset(uv_h5_paths,'uv')
        uv_dataloader = DataLoader(uv_dataset, batch_size=batch_uv_data, shuffle=True)
        dataloader_list.append(uv_dataloader)
        if (len(uv_dataloader) != 1):
            raise ValueError("uv_dataloader do not have the expected length of 1.")
        print("uv data batches: ", len(uv_dataloader))
    if C_p==1:
        p_dataset = DataPointsDataset(p_h5_paths,'p')
        p_dataloader = DataLoader(p_dataset, batch_size=batch_p_data, shuffle=True)
        dataloader_list.append(p_dataloader)
        if (len(p_dataloader) != 1):
            raise ValueError("p_dataloader do not have the expected length of 1.")
        print("p data batches: ", len(p_dataloader))
    
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
    # same batch for zip
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

    pinn_net = PINN_Net(layer_mat, net_type, lb, ub, device)
    print("Total parameter: ", count_params(pinn_net))
    pinn_net = pinn_net.to(device)

    # optimizer and lr decay or not
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
    
    
    for epoch in range(epochs):
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
            
            i_eqns = 0
            random_indices = torch.randperm(len(equation_dataset))[:batch_eq]
            random_data = [equation_dataset[i] for i in random_indices]
            x_eqns_bch, y_eqns_bch, t_eqns_bch = [torch.stack(tensors).requires_grad_(True).to(device) for tensors in zip(*random_data)]

            optimizer.zero_grad()

            mse_equation = pinn_net.eqns_mse(x_eqns_bch, y_eqns_bch, t_eqns_bch, **eq_dict)
            loss = coef_LoS * mse_LoS_predict + \
                   coef_Uvec * mse_Uvec_predict + \
                   coef_uv * mse_uv_predict + \
                   coef_p * mse_p_predict + \
                   coef_eqns * mse_equation

            loss.backward()
            optimizer.step()

            loss_all = loss.item()
            loss_eqns = mse_equation.item()

            # print("Epoch:", epoch + 1, 
            #       "Data_bch:", i_data + 1,
            #       "Eqns_bch:", i_eqns + 1,
            #       "Training Loss:", loss_all,
            #       "Data_LoS Loss:", loss_data_LoS,
            #       "Data_Uvec Loss:", loss_data_Uvec,
            #       "Data_p Loss:", loss_data_p,
            #       "Eqns Loss:", loss_eqns)
            # sys.stdout.flush()  # 刷新输出

            loss_all_epoch += loss_all
            loss_eqns_epoch += loss_eqns

        losses.append([loss_all_epoch, loss_data_LoS_epoch, loss_data_Uvec_epoch, 
                       loss_data_uv_epoch, loss_data_p_epoch, loss_eqns_epoch])
        scheduler_step()

        lr_now = optimizer.param_groups[0]['lr']
        lr.append(lr_now)

        # wandb.log to online record
        wandb_loss_log(epoch)
        
        if epoch in save_epoch_list:
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
                  "Eqns Loss:", loss_eqns_epoch)
            sys.stdout.flush()  # flush print
            
            valid_u, valid_v, valid_p = validation(pinn_net, real_path)
            wandb_valid_log(epoch)
            torch.save(pinn_net, work_path +'/PINN_pth_{}.pth'.format(epoch))
            torch.save(pinn_net.state_dict(), work_path +'/PINN_pth_{}.pt'.format(epoch))
            print("Model in epoch {} save to {}".format(epoch, work_path))
        else:
            continue
    print("Adam oK")

    if LBFGS:
        # use L-BFGS 
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
            
                i_eqns = 0
                random_indices = torch.randperm(len(equation_dataset))[:batch_eq]
                random_data = [equation_dataset[i] for i in random_indices]
                x_eqns_bch, y_eqns_bch, t_eqns_bch = [torch.stack(tensors).requires_grad_(True).to(device) for tensors in zip(*random_data)]
                
                # Define a dict to store loss value
                loss_values = {"loss_all": 0, "loss_data_LoS": 0, "loss_data_uv":0, "loss_data_p":0, "loss_eqns": 0, "LBFGS_No": 0}
                def closure():
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
                    # update the loss value dict
                    loss_values["LBFGS_No"] += 1
                    loss_values["loss_all"] = loss.item()
                    loss_values["loss_eqns"] = mse_equation.item()

                    return loss
                
                optimizer.step(closure)
                if (epoch + 1) % 10 == 0:
                    # watch NN parameter manually
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
                    sys.stdout.flush()

                loss_all_epoch += loss_values["loss_all"]
                loss_data_LoS_epoch += loss_values["loss_data_LoS"]
                loss_data_Uvec_epoch += loss_values["loss_data_Uvec"]
                loss_data_uv_epoch += loss_values["loss_data_uv"]
                loss_data_p_epoch += loss_values["loss_data_p"]
                loss_eqns_epoch += loss_values["loss_eqns"]

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
    # os.system("/usr/bin/shutdown") # auto power off

    
    