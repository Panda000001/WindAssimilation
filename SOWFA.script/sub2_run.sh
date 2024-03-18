#!/bin/bash
#SBATCH -p  hebhcnormal01
#SBATCH -N 1
##SBATCH -n 64
#SBATCH --exclusive
#SBATCH --ntasks-per-node=64

source /public/home/acg0a1h2j7/OpenFOAM/env.sh

cores=64

initializer=setFieldsABL
solver=ABLSolver
runNumber=1 # 1:0-20000s, 2:20000-20400s, 3: 20400-20500s

# echo "rm 0 and copy 0.original to 0"
# rm -rf 0
# cp -rf 0.original 0

cp system/controlDict.$runNumber system/controlDict

echo "Starting OpenFOAM job at: " $(date)
echo "using " $cores " cores"

# Run the flow field initializer (parallel)
if [ $runNumber -eq 1 ] 
  then
  #  mpirun -np $cores $initializer > log.$runNumber.$initializer 2>&1
  mpirun -np $cores $initializer -parallel > log.$runNumber.$initializer 2>&1
fi
# reconstructPar > log.reconstructPar 2>&1
# mpirun -np $cores reconstructPar -parallel > log.reconstructPar 2>&1
# Run the solver (parallel)
mpirun -np $cores $solver -parallel > log.$runNumber.$solver 2>&1
echo "Ending OpenFOAM job at: " $(date)

