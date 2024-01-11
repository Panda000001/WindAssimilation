# A Framework of Data Assimilation for Wind Flow Field by Physics-informed Neural Networks
This repository contains the source code for the research presented in the paper [A Framework of Data Assimilation for Wind Flow Field by Physics-informed Neural Networks](https://xxxxxxxxxxxxxxxxxxxxxxxxxxxx.)

## Overview
The main program, `main.py`, is the training process, in which the [wandb](https://wandb.ai/site) is included to search the optimized hyperparameter automatically. Physics-informed Neural Network (PINN) is defined in `pinn_model.py` under the framework of [PyTorch](https://pytorch.org/). All the requirements are included in `requirement.txt`. Four commonly available types of measurement data are supported: LoS wind speed, velocity vector, velocity component, and pressure. Given the turbulent nature of atmospheric boundary layer flow, the Reynolds-Averaged Navier-Stokes (RANS) equations are employed as the flow governing equations. The turbulence eddy viscosity, ${\nu _t}$, is directly predicted as an output variable of the PINN. The `pred_write.py` is used to reconstruct the wind flow field by a trained PINN. The reconstructed flow field data is written into `.h5` files. The script `transfer.py` is used for online deployment of the pre-trained PINN, assimilating real-time measured data.

## Test case
The test wind flow field is an atmospheric boundary layer flow simulated by [SOWFA (Simulator fOr Wind Farm Applications)](https://www.nrel.gov/wind/nwtc/sowfa.html). The flow field within the horizontal plane upstream of the wind turbine site is chosen to be the test area of the proposed framework. 
![Fig2_CFD_Result](./Visualization/Fig2_CFD_Result.jpg)
<img src="./Visualization/True100s.gif" alt="True100s" width="100" height="80" />
<img src="./Visualization/Case8Pred100s.gif" alt="Case8Pred100s" width="100" height="80" />
<img src="./Visualization/Case8Error100s.gif" alt="Case8Error100s" width="100" height="80" />

