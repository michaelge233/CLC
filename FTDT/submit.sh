#!/bin/bash

#SBATCH --job-name=meep
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -n 1
#SBATCH --time=23:59:00
#SBATCH --array=1-37
#SBATCH --output=slurm%A_%a.out 

source /home/mg2933/.bashrc
cd ${SLURM_SUBMIT_DIR}
module load Meep/1.26.0-foss-2020b

config=config.txt
angle=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
echo $angle
python3 RunFDTD.py $angle 10.0

