#!/bin/bash
#SBATCH --job-name=array_parallelized_deepsight
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=ryan.zurrin001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=100gb
#SBATCH -t 30-00:00
#SBATCH --output=array_%A-%a.out
#SBATCH --error=array_%A-%a.err
#SBATCH --array=1-17

. /etc/profile

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

python deep_sight_executor.py $SLURM_ARRAY_TASK_ID

echo "Finish Run"
echo "end time is `date`"