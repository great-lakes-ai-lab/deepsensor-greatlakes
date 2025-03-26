# Quick Guide to Using Slurm on U-M HPC

Jupyter Notebooks are great. But having to babysit them can be a pain, especially for long jobs. Follow the steps below to convert your notebook-based workflow into a batch script workflow.

## 1. Logging In

You have two options for getting a terminal interface on U-M HPC. 

### Via SSH
- **Open a Terminal:** On Windows, use an SSH client like PuTTY or Windows Terminal. MacOS and Linux have built-in terminals.
- **Run the SSH Command:** `ssh <uniqname>@greatlakes.arc-ts.umich.edu`
- Replace <uniqname> with your uniqname

### Via Open OnDemand
- **Access the Portal:** Go to https://greatlakes.arc-ts.umich.edu, as you normally do.
- **Log In:** Use your U-M credentials.
- **Launch a Terminal:** Use the "Clusters" menu and select "Great Lakes Shell Access."

## 2. Converting a Jupyter Notebook to a Python Script

To convert a Jupyter Notebook (.ipynb) file to a Python script (.py), follow these steps:

- Open Terminal and Navigate to the Notebook Directory:
```
cd path/to/notebook
```

Use nbconvert to Convert:
```
jupyter nbconvert --to script my_notebook.ipynb
```
This command will produce my_notebook.py in the same directory. Check the .py file for any errors or necessary modifications.


## 3. Creating a Batch Script
In nano, vi, or another text editor, create this file:

```
#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=01:00:00        # hh:mm:ss (reqest up to the maximum time for the partition you select)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --account=dannes1

# Load the necessary modules
module load python3.10-anaconda/2023.03
module load cuda/11.8.0 cudnn/11.8-v8.7.0

# Activate your virtual environment
source /home/uniqname/deepsensor_env_gpu/bin/activate

# Run your Python script
python my_notebook.py
```

## 4. Submitting the Slurm Script

Once you have written the script above, do the following:

- Once your script is ready: Name it `my_job_script.sh` or something like that.
- Submit the script with: `sbatch my_job_script.sh`

Running jobs via a Slurm script means the job will continue to run on the cluster even if you disconnect from the network. You can safely close your terminal or log out, knowing your job will still be processed.

For more information, see the documentation from ITS:

https://documentation.its.umich.edu/arc-hpc/slurm-user-guide
