#!/bin/bash
#SBATCH --job-name=lyrics_inf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=lyrics_inf.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ly1387@nyu.edu

module purge
module load anaconda3/2020.07

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

conda activate /scratch/ly1387/lyrics;
export PATH=/scratch/ly1387/lyrics/bin:$PATH;

cd /scratch/ly1387/projects/lyrics_generator;
python main.py
