#!/bin/bash
#SBATCH --job-name=deepsensor-train
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=20:00:00        # hh:mm:ss (reqest up to the maximum time for the partition you select)
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=dannes0

# Load the necessary modules
module load python3.10-anaconda/2023.03
module load cuda/11.8.0 cudnn/11.8-v8.7.0

# Activate your virtual environment
source /home/dannes/ds_env_gpu/bin/activate

# Run your Python script
python3 ds_train.py
