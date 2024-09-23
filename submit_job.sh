#!/bin/bash 
#SBATCH -C gpu 
#SBATCH -A als
#SBATCH -q debug
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --time=00:10:00
#SBATCH -o outputs/%x.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ShizhaoLu@lbl.gov

start=$(date +%s)
LOGDIR=${SCRATCH}/projects/mlex_dlsia_segmentation_prototype/logs
mkdir -p ${LOGDIR}

module load python

# conda environment in $HOME
# conda activate mgpu-seg-infer

# conda environment in $SCRATCH
conda activate /pscratch/sd/s/shizhaou/mgpu-seg-infer-scratch

args="${@}"

export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,nvtx,cublas,cudnn --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

set -x
srun bash -c "
    source export_env_vars.sh
    ${PROFILE_CMD} python src/segment_no_tiled.py ${args}
    "

end=$(date +%s)
echo "Total execution time: $(($end-$start)) seconds"