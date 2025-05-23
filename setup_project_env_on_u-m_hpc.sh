#!/bin/bash

# Load necessary HPC modules exactly 
module load python3.10-anaconda/2023.03
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0

# Activate your Python virtual environment
source /path/to/your/venv_name/bin/activate

