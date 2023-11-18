#!/bin/bash
#SBATCH --job-name=array_parallelized_deepsight
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=p.bendiksen001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=8gb
#SBATCH -t 14-6:30:00
#SBATCH --output=array_%A-%a.out
#SBATCH --error=array_%A-%a.err
#SBATCH --array=1-2

. /etc/profile

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate omama

python deepsight_executor_3d.py $SLURM_ARRAY_TASK_ID

echo "Finish Run"
