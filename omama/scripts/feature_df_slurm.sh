#!/bin/bash
#SBATCH --job-name=extract_features_and_pickle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ryan.zurrin001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH --ntasks=12 # Number of tasks
#SBATCH --cpus-per-task=1 # Number of CPUs per task
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=800G # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 30-00:00
#SBATCH --output=outputs/feature_extraction_%j.out
#SBATCH --error=outputs/feature_extraction_%j.err

. /etc/profile

echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

python /home/ryan.zurrin001/Projects/omama/_EXPERIMENTS/CS438/feature_extractor/feature_extractor.py /raid/mpsych/OMAMA/DATA/whitelists/final_2d_dataset.txt --cores 12 --store_images
