#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 50:59:00
#SBATCH --mem-per-cpu=20G

conda activate regio_ch
# run the code
sh submit_file
