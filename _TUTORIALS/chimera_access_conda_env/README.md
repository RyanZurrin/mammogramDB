# Chimera Account Access and Conda Env Creation.
This directory instructs researchers, having access to the Chimera high performance computing cluster, on how to create a reproducible python environment that ensures fast access of GPUs for deep learning applications.

## Account Access
Once an account has been created for you on the chimera cluster your login will be the same as your UMB email credentials (same username and password).
1. For ssh access:
    ```
    $ ssh -tt user.name@chimera.umb.edu 
    ```
You will need a VPN for off campus access. Please visit this web page for information about VPN access: 

[https://www.umb.edu/it/vpn](https://www.umb.edu/it/vpn)

Once you ssh into your chimera account, you will be at the head node of the cluster, but the focus is to run jobs from one of the DGXA100 (GPU related) compute nodes, otherwise known as Chimera12 and Chimera13. 

2. Start a job requesting either of the DGX nodes: 
    ```
    srun --pty -t <HH:MM:SS> --export=NONE --ntasks=4 -p haehn -q haehn_unlim --gres=gpu:A100:1 --mem=<MEM_AMOUNT>  /bin/bash
    ```
    where t (time) and mem (memory requested) are arbitrarily specified by you;  <em> please ensure to specify a reasonable time frame (e.g. under 24 hours) as well as memory amount (e.g. '128G', though this may be excessive for your needs), and note that not specifying the t value defaults to the max time permitted (which will saturate node access) <em>
    
    Alternatively, if you care to access the Chimera12 compute node specifically, run:
    ```
     srun --pty -t <HH:MM:SS> --export=NONE --ntasks=4 -p haehn -q haehn_unlim -w chimera12 --gres=gpu:A100:1 --mem=<MEM_AMOUNT>  /bin/bash
     ```
    Options arguments for your parallel job request can be found [here](https://slurm.schedmd.com/srun.html) 
    
    The first example provides the following arguments when starting the job: 24 hour access (24:00:00), ensuring not to export any environment variables from the head node profile, requesting 24 processors, priority account privilege (haehn), access of 1 GPU (gpu:A100:1), and access to 512G of memory.
    
3. Source the profile of the DGX node: 
    ```
    source /etc/profile
    ```
    The reason we have the requirement to source the profile once in the compute node is because the DGX nodes have a different OS (ubuntu) than the rest of the cluster (Rocky Linux). 
    
## Miniconda for Linux Installation
<em>Please ensure that the following steps are taken from within a compute node only</em>

Conda is an open source package management system and environment management system that runs on Windows, macOS and Linux.
The Conda *Read The Docs* website gives information on how to download and install Anaconda/Miniconda for Linux: 

[https://conda.io/projects/conda/en/latest/user-guide/install/linux.html](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)  

1. Download the installer:
    ```
    $ wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
    ```
2. Verify your installer hashes.

    To ensure your file originates from the developer we recommend you check it hashes succesfully:

     ```
    $ sha256sum Miniconda3-py39_4.9.2-Linux-x86_64.sh
    ```
3. Install Miniconda.
    
    In your terminal window, run:

    ```
    $ bash Miniconda3-latest-Linux-x86_64.sh
    ```
    and follow the prompts of the installer screens. To ensure changes take effect, close and then re-open your terminal window.
4. Test your installation.
    
    In your terminal window, run the command:

    ```
    $ conda list
    ```
    A list of installed packages should appear if installation was succesful.

## Conda Env Creation
<em>Please ensure that the following steps are taken from within a compute node only</em>

Next, you will need to set up a conda environment; the provided O YAML file ensures a standardized conda environment for GPU accessible tensoflow (and related dependencies) use.

1. Create directory for environment.

    Please make sure the name of the directory coincides with the name of the YAML file:

    ```
    mkdir /home/user.name/miniconda3/envs/TF25dcm
    ```
2. Copy YAML file to new directory.

    Replace the "prefix" field located at the end of the provided YAML file to coincide with the path to your newly created directory. 
    Then, open another terminal instance and copy the YAML file from your local machine (at whatever pwd you are in) to your newly created environment directory (FYI: anything copied into the chimera head home directory will be visible from other nodes):

    ```
    $ scp TF25dcm.yml user.name@chimera.umb.edu:/home/user.name/miniconda3/envs/TF25dcm
    ```

3. Create Conda env.

    To create your conda environment by means of the Yaml file, run the following:
    ```
    $ cd /home/user.name/miniconda3/envs/TF25dcm
    ```
    ```
    $ conda env create -f O.yml
    ```
    If this command fails please attempt a repeat by replacing the '.yml' file extension with '.yaml'
    
    To activate your newly created environment, run the following:
    ```
    $ conda activate O
    ```

## Test Python for GPU access.
Now that your conda environment is activated, go ahead and start a Python interpreter program: 
```
$ python
```
and run the following Python code:
```
import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")

# Check that the print value is greater than 0:
print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
```
