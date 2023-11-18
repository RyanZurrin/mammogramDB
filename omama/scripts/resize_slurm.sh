#!/bin/bash
#SBATCH --job-name=resized_2d
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ryan.zurrin001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH --ntasks=100 # Number of tasks
#SBATCH --cpus-per-task=1 # Number of CPUs per task
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=300gb # Memory for all CPUs
#SBATCH -t 30-00:00
#SBATCH --output=2d_resized_%j.out
#SBATCH --error=2d_resized_%j.err

. /etc/profile

echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

python resize_and_rescale_multi.py -i /raid/mpsych/OMAMA/DATA/data/2d/ -o /raid/mpsych/OMAMA/DATA/data/2d_resized_512/ -w 512 -ht 512 -p 100
