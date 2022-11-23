#!/bin/bash
#SBATCH --job-name=lyrics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=lyrics.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ly1387@nyu.edu

module purge
module load anaconda3/2020.07

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

conda activate /scratch/ly1387/lyrics;
export PATH=/scratch/ly1387/lyrics/bin:$PATH;

cd /scratch/ly1387/projects/lyrics_generator;
python train_bert_nsp.py
