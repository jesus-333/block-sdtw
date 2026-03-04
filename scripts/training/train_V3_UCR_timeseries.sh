#!/bin/sh

#SBATCH --job-name="UCR_analysis_block_SDTW"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=0-00:20:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./Results/hpc_log/std_output_%x_%j.txt
#SBATCH --error=./Results/hpc_log/other_output_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

echo "---------------------------------------------------"
echo $CONDA_DEFAULT_ENV
echo "---------------------------------------------------"
conda init
conda activate jesus-hpc
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
echo $CONDA_DEFAULT_ENV
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

which python

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Launch FL Training

srun python ./train_V3_UCR_timeseries.py

