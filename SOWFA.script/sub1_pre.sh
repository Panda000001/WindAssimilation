#!/bin/bash
#SBATCH -p  hebhcnormal01
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --exclusive
#SBATCH --ntasks-per-node=64

source /public/home/acg0a1h2j7/OpenFOAM/env.sh

# User Input.
# OpenFOAMversion=2.4.x-central   # OpenFOAM version
startTime=0                     # Start time
updateBCType=0                  # Boolean for whether or not the boundary condition types will be updated over 
                                #    what is in the initial conditions files. Leave it 0 for precursors
inflowDir='cyclic'              # For inflow/outflow cases, specify the inflow direction.  Choices are 'west',
                                #   'east', 'south', 'west', 'southWest', 'northWest', 'southEast', and
                                #   'northEast'.  There is a 'cyclic' option too in case you need to change
                                #    back to cyclic or simply update the number of boundary face entries.
parallel=1                      # Boolean for whether or not the preprocessing is run in parallel.
cores=64                         # Enter the number of cores you will preprocess on.

refinementLevels=0              # If you want to refine the mesh locally for any reason, or if you are making
                                # a uniform resolution mesh that is so large that you need to build it in serial
                                # at lower resolution and then globally refine, set the number of refinement
                                # levels here.  See the refineMeshSerial and refineMeshParallel functions to 
                                # understand what they do.  The serial version runs topoSet and refineHexMesh, 
                                # so you need to provide system/topoSetDict.local.N files where N corresponds
                                # to the refinement level (i.e., if you are doing nested local refinement boxes.
                                # In most cases, though, you probably will not be refining, so keep this set to 
                                # 0.




# Define some functions for mesh refinement.
# Local refinement performed on one core.
refineMeshLocal()
{
   i=$1
   while [ $i -ge 1 ]
   do
      echo "   -Performing level $i local refinement with topoSet/refineHexMesh"
      echo "      *selecting cells to refine..."
      topoSet -dict system/topoSetDict.local.$i > log.topoSet.local.$i 2>&1

      echo "      *refining cells..."
      refineHexMesh local -overwrite > log.refineHexMesh.local.$i 2>&1

      let i=i-1
   done
}

# Global refinement performed in parallel.
refineMeshGlobal()
{
   i=1
   while [ $i -le $1 ]
   do
      echo "   -Performing level $i global refinement with refineMesh"
      echo "      *refining cells..."
      mpirun -np $cores refineMesh -all -overwrite > log.refineMesh.global.$i 2>&1

      let i=i+1
   done
}


# If running in parallel, cd to job launch directory
if [ $parallel -eq 1 ]
    then
    #  cd $PBS_O_WORKDIR
    cd $SLURM_SUBMIT_DIR
fi

echo pwd
# Source the bash profile and then call the appropriate OpenFOAM version function
# so that all the modules and environment variables get set.
# source $HOME/.bash_profile
# OpenFOAM-$OpenFOAMversion


# Copy the controlDict.1 (assuming this is the one the actual solver will start
# out with) to controlDict.  OpenFOAM reads "controlDict", not the numbered ones.
echo "copy controlDict.1 to controlDict"
cp system/controlDict.1 system/controlDict


# Copy the "clean" .original initial fields to a working copy.  OpenFOAM does not
# read the ".original" initial fields--that's why they remain clean.

echo "rm startTime(0) and copy 0.original to startTime(0)"
rm -rf $startTime
cp -rf $startTime.original $startTime


# Build the mesh.
echo "Build the mesh by blockMesh."
cp constant/polyMesh/blockMeshDict ./
rm -rf constant/polyMesh/*
mv ./blockMeshDict constant/polyMesh
blockMesh > log.blockMesh 2>&1


# The initial fields come from the precursor which is periodic on all sides.  The turbine
# case has inflow and outflow.  Call the changeDictionary utility to make the south and
# north sides inflow and outflow.
if [ $updateBCType -eq 1 ]
   then
   changeDictionary -dict system/changeDictionaryDict.updateBCs.$inflowDir -time $startTime -enableFunctionEntries > log.changeDictionary.updateBCs.$inflowDir.1 2>&1
fi


# Do serial local refinement
# refineMeshLocal $refinementLevels


# If running in parallel from this point forward, then do the following:
if [ $parallel -eq 1 ]
   then
   # Decompose the mesh and solution files (serial)
   decomposePar -cellDist -force > log.decomposePar 2>&1

   # Check the mesh
   # mpirun -np $cores checkMesh > log.checkMesh.1 2>&1
   mpirun -np $cores checkMesh -parallel > log.checkMesh.1 2>&1
   # Perform global refinement to desired resolution.
   # refineMeshGlobal $refinementLevels

   # The mesh got globally refined, but the solution file did not, so
   # the boundary fields may not have the correct number of entries.
   # Use the changeDictionary utility to overwrite the spatially varying
   # boundary data to a uniform single value.
   if [ $updateBCType -eq 1 ]
      then
      mpirun -np $cores changeDictionary -dict system/changeDictionaryDict.updateBCs.$inflowDir -time $startTime -enableFunctionEntries > log.changeDictionary.updateBCs.$inflowDir.1 2>&1
   fi

   # Renumber the mesh for better matrix solver performance.
   # mpirun -np $cores renumberMesh -overwrite > log.renumberMesh 2>&1
   mpirun -np $cores renumberMesh -parallel -overwrite > log.renumberMesh 2>&1
   # Do one last check on the mesh.
   # mpirun -np $cores checkMesh > log.checkMesh.3 2>&1
   mpirun -np $cores checkMesh -parallel > log.checkMesh.3 2>&1


# Otherwise, run in serial as follows:
else
   # Renumber the mesh.
   echo "   -Renumbering the mesh with renumberMesh..."
   renumberMesh -overwrite > log.renumberMesh 2>&1

   # Decompose the mesh and solution files (serial)
   echo "   -Decomposing the domain with decomposePar..."
   decomposePar -cellDist -force > log.decomposePar 2>&1

   # Check the mesh.
   echo "   -Checking the mesh with checkMesh..."
   checkMesh > log.checkMesh.1 2>&1
   
   echo " Pre All Done."
fi
