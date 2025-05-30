# Environment Setup for DeepSensor

This guide outlines procedures to configure the DeepSensor environment on the Great Lakes High-Performance Computing (HPC) platform at the University of Michigan for both GPU and non-GPU use. It differentiates between the initial setup of the environment and later use, as well as providing instructions for working with and without the GitHub repository.

## Table of Contents
- [Quick Start: Automated Setup Script](#quick-start-on-u-m-great-lakes-hpc-automated-setup-script)
- [Non-GPU Environment Setup](#cpu-only-environment-setup)
- [GPU Environment Setup](#gpu-environment-setup-with-pytorch)
  - [Setting Up for the First Time](#setting-up-for-the-first-time)
  - [Using the Environment Thereafter](#using-the-environment-thereafter)
  - [PyTorch GPU Support](#pytorch-gpu-support)
- [Working with the GitHub Repository](#working-with-the-github-repository)
- [Setting Up a Jupyter Notebook Session on Great Lakes HPC](#setting-up-a-jupyter-notebook-session-on-great-lakes-hpc)
- [Slurm Script for Command Line GPU Jobs](#slurm-script-for-command-line-gpu-jobs)
- [GPU Environment Setup for Google Vertex AI](#gpu-only-environment-setup-for-deepsensor-on-vertex-ai)

## Quick Start on U-M Great Lakes HPC: Automated Setup Script
If you want to quickly set up a GPU-enabled environment on U-M Great Lakes HPC, you can use the provided Bash script `utils/create_new_deepsensor_env_on_GL.sh`. This script automates the following tasks:

- Loads the necessary system modules for GPU support.
- Creates and activates a virtual environment.
- Installs required Python packages, including PyTorch with GPU support and DeepSensor.
- Adds the virtual environment as a Jupyter kernel for ease of use.

### Using the Script
1. Clone or download the repository containing the script.
2. Run the script in a terminal session on Great Lakes HPC:
```bash utils/create_new_deepsensor_env_on_GL.sh```
3. Follow the prompts to name your environment and confirm setup steps.

Note: This script is tailored specifically for the Great Lakes HPC system and may not work on other platforms without modifications.

### Tutorial Video
[![Screenshot](https://github.com/user-attachments/assets/a92732a3-9bd1-44b0-8f17-99bf3fd37614)](https://youtu.be/bCGabxyTyYc)

### (Optional) Install Tools In This Repository
After installing the necessary dependencies above, you can import this `deepsensor-greatlakes` repo as a package by running the following:
```bash
pip install .
```
This command should be run from within the repository directory (the root of this repo), where the `setup.py` file is located. 
If you wish to develop the tools in this repository, install the package in editable mode:
```bash
pip install -e .
```

For detailed manual setup instructions, refer to the sections below.

## CPU-only Environment Setup
To set up DeepSensor for non-GPU use, follow these instructions:

### Setting Up for the First Time

1. **Load the Anaconda Module:**

    If you are working on Great Lakes HPC, load the Anaconda module which provides a base Python installation:

    ```bash
    module load python3.10-anaconda/2023.03
    ```

2. **Create and Activate a Virtual Environment:**

    Create a new virtual environment in your home directory or another location where you have write permissions:

    ```bash
    python -m venv ~/deepsensor_env
    ```

    Activate the virtual environment:

    ```bash
    source ~/deepsensor_env/bin/activate
    ```

3. **Install DeepSensor and PyTorch:**

    Install the `deepsensor` package from PyPI, which is necessary for running the CPU-only version. If you need PyTorch for deep learning modeling, install that as well:

    ```bash
    pip install deepsensor
    pip install torch
    ```

    Note: PyTorch will default to the CPU-only version if you don't have the CUDA toolkit installed.

4. **Additional Setup for Jupyter Notebooks:**

   If you would like to use this environment in a Jupyter notebook, you have to add the virtual environment as a Jupyter kernel:

   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=deepsensor_env --display-name "Python (deepsensor_env)"
   ```
   If successful, you will see `Installed kernelspec` message on the command line. 

### Using the Environment Thereafter

After the first-time setup, you can use your environment on subsequent logins to the HPC platform:

1. **Activate Your Virtual Environment:**

    Before starting your work, activate the virtual environment:

    ```bash
    source ~/deepsensor_env/bin/activate
    ```

2. **Work on Your DeepSensor Project:**

    Run your Python scripts or start a Jupyter Notebook that utilizes the `deepsensor` environment.

3. **Deactivate Your Virtual Environment (Optional):**

    Once you're done, you can deactivate the virtual environment to return to the base environment:

    ```bash
    deactivate
    ```

### For Jupyter Notebook Users

After you have added your virtual environment as a Jupyter kernel (see above), you can carry out the following steps to use your virtual environment as a Jupyter notebook. To use a non-GPU environment with Jupyter Notebooks on U-M HPC:

1. Navigate to the Great Lakes HPC Jupyter service and fill out the job submission form as needed.
2. In the "Module commands" field, only specify the Anaconda Python module, as no GPU resources are requested.
3. In the "Source this setup file" field, specify the path to a script to automate the loading of the module and activation of the environment (e.g. `source /full-path-to/deepsensor_env/bin/activate`)

When your Jupyter Notebook session starts, you can proceed to work with your notebooks that leverage the DeepSensor package without GPU support. When starting a new notebook, select the kernel that has the name of your 
virtual environment in parentheses (see image below for example). 

![image](https://github.com/CIGLR-ai-lab/GreatLakes-TempSensors/assets/11757453/2b4c7ee5-29f7-4116-b08a-adad1e48222e)

If you are working with an existing Jupyter notebook, you can select `Kernel -> change kernel` from the task bar and select the kernel that you wish to use, corresponding to the virtual environment in which you want to work. 

## GPU Environment Setup with Pytorch

### Setting Up for the First Time
To initially configure the environment for GPU-accelerated work:

1. Load the necessary system modules (assuming CUDA v11.8.0): 

    ```bash
    module load python3.10-anaconda/2023.03
    module load cuda/11.8.0
    module load cudnn/11.8-v8.7.0
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv ~/deepsensor_env_gpu  # Create a new environment
    source ~/deepsensor_env_gpu/bin/activate  # Activate the environment
    ```

3. If using PyTorch with GPU (instructions from https://pytorch.org/. Your case may differ):

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip3 install deepsensor
    ```
    
4. Additional Setup for Jupyter Notebooks:

   If you would like to use this environment in a Jupyter notebook, you have to add the virtual environment as a Jupyter kernel:
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=deepsensor_env_gpu --display-name "Python (deepsensor_env_gpu)"
   ```
   If successful, you will see `Installed kernelspec` message on the command line. 

### Using the Environment Thereafter
Once the initial setup is complete, you can activate your environment with:

```bash
source ~/deepsensor_env_gpu/bin/activate
```

And deactivate it when you're finished:

```bash
deactivate
```

## Working with the GitHub Repository
If you choose to utilize specific functions and notebooks from the DeepSensor GitHub repository:

```bash
git clone https://github.com/your-username/deepsensor.git  # Clone the repository
cd deepsensor  # Navigate to the repository directory
# Then proceed with the GPU or non-GPU environment setup as detailed above.
```

To use DeepSensor without incorporating additional functions from the GitHub repository, simply skip the previous step. 

## Setting Up a Jupyter Notebook Session on Great Lakes HPC

When launching a Jupyter Notebook on Great Lakes HPC, fill out the web form with the following selections:

| **Field**                 | **Details**                                                                                                     | **Example / Notes**                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **Anaconda Python Module** | Choose the version of Python for your Jupyter Notebook server.                                                  | `python3.10-anaconda/2023.03`                                                                         |
| **Slurm Account**          | Enter the Slurm account identifier.                                                                             | `dannes0`                                                                                            |
| **Partition**              | Select the partition for computation. Use `gpu` for GPU-accelerated tasks.                                     | `gpu`                                                                                                |
| **Number of Hours**        | Specify the job duration in hours.                                                                              | `1` hour                                                                                             |
| **Number of Cores**        | Indicate the number of CPU cores your job will use. Use `1` for non-parallelized jobs.                          | `1` core                                                                                             |
| **Memory**                 | Request the memory needed for the job.                                                                          | `40 GB`                                                                                              |
| **Number of GPUs**         | Specify the number of GPUs required for the job. Usually `1` unless designed for multi-GPU processes.           | `1`                                                                                                  |
| **GPU Compute Mode**       | Specify GPU usage mode: `shared` if multiple processes share the GPU, or `exclusive` if not specified.          | `shared`                                                                                            |
| **Software Licenses**      | Request any specific software licenses required by your job.                                                    | Leave blank if not applicable.                                                                       |
| **Module Commands**        | List modules to be loaded before starting the session.                                                          | `load cuda/11.8.0 cudnn/11.8-v8.7.0`                                                          |
| **Source Setup File**      | Provide the path to the file that activates your virtual environment.                                            | `/home/uniqname/deepsensor_env_gpu/bin/activate`                                                     |

Please note that these instructions assume you have already created and configured the `deepsensor_env_gpu` virtual environment as per the [GPU Environment Setup](#gpu-environment-setup) section.

Once your Jupyter Notebook session has started, make sure to select the kernel that corresponds to your virtual environment, as shown in the image below:

![image](https://github.com/CIGLR-ai-lab/GreatLakes-TempSensors/assets/11757453/489eb463-fa26-4104-88d8-6ee3a62ec881)

If you are working with an existing Jupyter notebook, you can select `Kernel -> change kernel` from the task bar and select the kernel that you wish to use, corresponding to the virtual environment in which you want to work. 

## Slurm Script for Command Line GPU Jobs
Use the following template for submitting GPU jobs via the command line:

```bash
#!/bin/bash
#SBATCH --job-name=deepsensor_gpu_test
#SBATCH --mail-type=END,FAIL
#SBATCH --account=account_name
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=deepsensor_gpu_test_%j.log

echo "Starting job on: $(hostname)"
echo "Activating virtual environment..."

module load python3.10-anaconda/2023.03
module load cuda/11.8.0
module load cudnn/11.8-v8.7.0
source ~/deepsensor_env_gpu/bin/activate

echo "Running DeepSensor with GPU support..."
python ~/DeepSensor/deepsensor_gpu_test.py

echo "Job completed on: $(date)"
```

Remember to replace `account_name` with your actual Slurm account name and adjust resource requests as appropriate.

In the above case, the file `deepsensor_gpu_test.py` is a simple import and GPU test:

```
import logging
logging.captureWarnings(True)

import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader, construct_circ_time_ds
from deepsensor.model import ConvNP
from deepsensor.train import set_gpu_default_device

# Run on GPU if available by setting GPU as default device
set_gpu_default_device()
```

If the above code returns nothing, then the GPU has been detected.

## GPU-Only Environment Setup for DeepSensor on Vertex AI
The instructions below should help you set up a GPU-enabled environment for DeepSensor on our Google Vertex AI Workbench (JupyterLab). The setup assumes that Python 3.10.16 is available by default on the Vertex AI instance.

Step-by-Step Setup
### 1. Open a Terminal in JupyterLab
Open a terminal window within the JupyterLab interface on Vertex AI.

### 2. Install/Upgrade pip (Optional)
Ensure you have the latest version of pip:

```bash
pip install --upgrade pip
```

### 3. Install PyTorch with CUDA Support
To install PyTorch with CUDA 11.8 support, use the following command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install DeepSensor
Install the DeepSensor library from PyPI:

```bash
pip install deepsensor
```

### 5. Install Additional Packages
If you need any additional packages (for handling large datasets, plotting, etc.), you can install them with:

```bash
pip install xarray zarr numpy pandas matplotlib seaborn scikit-learn dask gcsfs
```

### 6. Install ipykernel and Register Your Environment as a Jupyter Kernel
To use the environment in JupyterLab, register it as a kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name=deepsensor_env_gpu --display-name "Python (deepsensor_env_gpu)"
```

### 7. Verify CUDA and PyTorch Installation
Run the following Python script in a Jupyter notebook to check if CUDA and PyTorch are working properly:

```python
import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check CUDA version
print("CUDA version:", torch.version.cuda)

# Check PyTorch version
print("PyTorch version:", torch.__version__)
```

If everything is set up correctly, you should see that CUDA is available, and it will print the correct PyTorch and CUDA versions.

### 8. Verify GPU Availability with DeepSensor Code
Run the following to test GPU functionality in DeepSensor. If all works correctly, this should return blank:

```python
import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader, construct_circ_time_ds
from deepsensor.model import ConvNP
from deepsensor.train import set_gpu_default_device

# Run on GPU if available by setting GPU as default device
set_gpu_default_device()
```
If all of that works, your DeepSensor GPU environment should be ready to go! 

---
