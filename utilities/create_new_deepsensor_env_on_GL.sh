#!/bin/bash

# --------------------------------------------------------
# Script: create_new_deepsensor_env_on_GL.sh
# Purpose: Automates the setup of a Python virtual environment
#          configured for DeepSensor on the U-M Great Lakes HPC,
#          including integration with Jupyter.
#
# Assumptions:
# - This script is designed specifically for the U-M Great Lakes HPC.
# - You have access to Great Lakes HPC and the necessary permissions.
# - Modules "python3.10-anaconda", "cuda", and "cudnn" are available.
# - CUDA version 11.8 and cuDNN version 8.7.0 are required.
#
# Usage:
# 1. Save this script to your Great Lakes HPC environment.
# 2. Run it using the command: bash create_new_deepsensor_env_on_GL.sh
# 3. Follow the prompts to specify the virtual environment name.
#    (Default name: deepsensor_env_gpu)
#
# Outputs:
# - A Python virtual environment configured for PyTorch, DeepSensor,
#   and Jupyter kernel integration.
# - Environment location: ~/deepsensor_env_gpu (or specified name).
# - A Jupyter kernel added for the new environment, visible as
#   "Python (your_env_name)" in Jupyter Notebook or JupyterLab.
#
# Post-Setup:
# - Activate the environment using:
#   source ~/deepsensor_env_gpu/bin/activate
# - Deactivate it using:
#   deactivate
# - To use the Jupyter kernel, open Jupyter and select the
#   appropriate kernel from the dropdown.
#
# Notes:
# - This script installs dependencies via pip, including GPU-enabled
#   PyTorch compatible with CUDA 11.8.
#
# --------------------------------------------------------

# Prompt for the virtual environment name
read -p "Enter the name for your virtual environment: " VENV_NAME

# Default name if none provided
VENV_NAME=${VENV_NAME:-deepsensor_env_gpu}

# Modules to load (customize as needed)
MODULES=("python3.10-anaconda/2023.03" "cuda/11.8.0" "cudnn/11.8-v8.7.0")

echo "Loading necessary modules..."
for MODULE in "${MODULES[@]}"; do
    module load "$MODULE"
done

# Create the virtual environment
VENV_DIR=~/"$VENV_NAME"
echo "Creating virtual environment at $VENV_DIR..."
python -m venv "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install required Python packages
echo "Installing PyTorch and other dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepsensor ipykernel

# Add virtual environment as a Jupyter kernel
echo "Adding the virtual environment as a Jupyter kernel..."
python -m ipykernel install --user --name="$VENV_NAME" --display-name "Python ($VENV_NAME)"

# Deactivate the environment
deactivate

echo "Setup complete! To use the environment, activate it with:"
echo "source $VENV_DIR/bin/activate"
