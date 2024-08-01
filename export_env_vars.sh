# required environment variables for multi-node multi-gpu for each gpu
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT=29500 # default from torch launcher

# required when using numba njit parallel which is multi-threaded

# also note from openblas document:
# If your application is already multi-threaded, it will conflict with OpenBLAS multi-threading. Thus, 
# you must set OpenBLAS to use single thread in any of the following ways

export OPENBLAS_NUM_THREADS=1
