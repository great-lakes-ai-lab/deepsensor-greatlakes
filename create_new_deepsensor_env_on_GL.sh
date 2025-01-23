#!/bin/bash

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
