#!/bin/bash
#SBATCH --job-name=sdl
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera13
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=115gb
#SBATCH --gres=gpu:A100:1
#SBATCH -t 30-00:00
#SBATCH --output=output/sdl_%A-%a.out
#SBATCH --error=output/sdl_%A-%a.err

##. /etc/profile,

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

python odl.py 3 ASTAR custom $SLURM_JOB_ID 1 50bins_no_gaussian

echo "Finish Run"
echo "end time is `date`"
