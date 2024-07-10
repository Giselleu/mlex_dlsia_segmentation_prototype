#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -A als
#SBATCH -q regular
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node 4
#SBATCH --time=00:10:00
#SBATCH -J seg-inference-DDP
#SBATCH -o %x-%j.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ShizhaoLu@lbl.gov

module load python
conda activate mgpu-seg-infer

args="${@}"

export MASTER_ADDR=$(hostname)
# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun bash -c "
    source export_DDP_vars.sh
    python src/segment_no_tiled.py example_yamls/example_tunet.yaml ${args}
    "

