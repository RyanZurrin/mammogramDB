#!/bin/bash
#SBATCH --job-name=array_parallelized_deepsight
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=ryan.zurrin001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=8gb
#SBATCH -t 14-6:30:00
#SBATCH --output=deepsight_out_msg/array_%A-%a.out
#SBATCH --error=deepsight_out_msg/array_%A-%a.err
#SBATCH --array=1-2

. /etc/profile

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

python deepsight_executor_3d_toy.py $SLURM_ARRAY_TASK_ID

echo "Finish Run"
