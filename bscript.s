#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=dl_iou_03_lr_10e-5_depth_1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/slurm_dl_iou_03_lr_10e-5_depth_1_%j.out

. ~/.bashrc
module load anaconda3/5.3.1 

source activate sdc3
#conda install -n cvproj nb_conda_kernels

cd /scratch/jm7519/deeplearning/DL-TopDownRoad/

python --version
python train.py --iou 0.3 --lr 10e-5 --depth 1
