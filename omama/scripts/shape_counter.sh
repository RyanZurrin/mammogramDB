#!/bin/bash
#SBATCH --job-name=extract_and_save_3d_pixel_arrays
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ryan.zurrin001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH --ntasks=100 # Number of tasks
#SBATCH --cpus-per-task=1 # Number of CPUs per task
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=300gb # Memory for all CPUs
#SBATCH -t 30-00:00
#SBATCH --output=eas_3d_pixel_arrays_%j.out
#SBATCH --error=eas_3d_pixel_arrays_%j.err

. /etc/profile

echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

python get_shapes_multi_cpu.py -i /raid/mpsych/OMAMA/DATA/data/2d/ -t 64 -o /raid/mpsych/OMAMA/DATA/data/2d/shapes_count2.txt
