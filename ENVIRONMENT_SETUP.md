# Environment Setup for DeepSensor

This guide outlines procedures to configure the DeepSensor environment on the Great Lakes High-Performance Computing (HPC) platform at the University of Michigan for both GPU and non-GPU use. It differentiates between the initial setup of the environment and later use, as well as providing instructions for working with and without the GitHub repository.

## Table of Contents
- [Non-GPU Environment Setup](#cpu-only-environment-setup)
- [GPU Environment Setup](#gpu-environment-setup-with-pytorch)
  - [Setting Up for the First Time](#setting-up-for-the-first-time)
  - [Using the Environment Thereafter](#using-the-environment-thereafter)
  - [PyTorch GPU Support](#pytorch-gpu-support)
- [Working with the GitHub Repository](#working-with-the-github-repository)
- [Setting Up a Jupyter Notebook Session on Great Lakes HPC](#setting-up-a-jupyter-notebook-session-on-great-lakes-hpc)
- [Slurm Script for Command Line GPU Jobs](#slurm-script-for-command-line-gpu-jobs)

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

1. **Anaconda Python Module:**
   - Choose `python3.10-anaconda/2023.03` to define the version of Python your Jupyter Notebook server will use.

2. **Slurm Account:**
   - Enter the Slurm account identifier, for example: `dannes0`.

3. **Partition:**
   - Select the `gpu` partition for GPU-accelerated computation.

4. **Number of Hours:**
   - Specify the job duration, e.g., `1` hour.

5. **Number of Cores:**
   - Indicate the number of CPU cores your job will use, suggesting `1` if your program is not parallelized.

6. **Memory:**
   - Request the desired memory for the job, e.g., `40 GB`.

7. **Number of GPUs:**
   - Specify the number of GPUs needed, usually `1` unless your process is specifically designed for multiple GPUs.

8. **GPU Compute Mode:**
   - Set to shared if multiple processes will share the GPU, exclusive if not specified.

9. **Software Licenses (if required):**
   - Request any specific software licenses that your job requires.

10. **Module Commands:**
    - Enter modules to be loaded before starting the session:
      ```
      load cuda/11.8.0 cudnn/11.8-v8.7.0
      ```
    - Ensure to include all necessary modules.

11. **Source Setup File:**
    - Provide the path to the setup file that activates your virtual environment:
      ```
      /home/uniqname/deepsensor_env_gpu/bin/activate
      ```
    - This file should contain activation commands and be executable.

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

---
