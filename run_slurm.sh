NUM_GPU_NODE=1
NUM_GPUS=1
CPUS_PER_GPU=32
INFERENCE_BATCH_SIZE=128

QLTY_WINDOW_SIZE=64
QLTY_STEP_SIZE=32

EXP_ID=161650

MODEL_CONFIG=TUNet-win-${QLTY_WINDOW_SIZE}-step-${QLTY_STEP_SIZE}
CONFIG=${MODEL_CONFIG}-ngpu-${NUM_GPUS}-ncpu-${CPUS_PER_GPU}-bs-${INFERENCE_BATCH_SIZE}

TRIAL=0
JOB_DUP_NAME=seg-infer-${CONFIG}-${TRIAL}
# JOB_DUP_NAME=seg-infer-64-original-${TRIAL}


# Enable profiling: 0 run timing, not save seg results tiffs
# Enable profiling: 1 run profiling, only runs a small number of frames, single gpu

# for profiling
ENABLE_PROFILING=1 PROFILE_OUTPUT=seg-infer-${CONFIG} sbatch -N ${NUM_GPU_NODE} -J seg-infer-${CONFIG} --cpus-per-task ${CPUS_PER_GPU} submit_job.sh --yaml_path example_yamls/example_tunet.yaml --slurm_run_name ${CONFIG} --model_config ${MODEL_CONFIG}-${EXP_ID} --inference_batch_size ${INFERENCE_BATCH_SIZE} --qlty_window_size ${QLTY_WINDOW_SIZE} --qlty_step_size ${QLTY_STEP_SIZE}

# for timing
# ENABLE_PROFILING=0 sbatch -N ${NUM_GPU_NODE} -J ${JOB_DUP_NAME} --cpus-per-task ${CPUS_PER_GPU} submit_job.sh --yaml_path example_yamls/example_tunet.yaml --slurm_run_name ${CONFIG} --model_config ${MODEL_CONFIG}-${EXP_ID} --inference_batch_size ${INFERENCE_BATCH_SIZE} --qlty_window_size ${QLTY_WINDOW_SIZE} --qlty_step_size ${QLTY_STEP_SIZE}

# for testing purposes
# ENABLE_PROFILING=1 PROFILE_OUTPUT=seg-infer-${QLTY_WINDOW_SIZE}-new-test-test sbatch -N ${NUM_GPU_NODE} -J seg-infer-${QLTY_WINDOW_SIZE}-original-new --cpus-per-task ${CPUS_PER_GPU} submit_job.sh --yaml_path example_yamls/example_tunet.yaml --slurm_run_name seg-infer-${QLTY_WINDOW_SIZE}-numba --model_config ${MODEL_CONFIG}-${EXP_ID} --inference_batch_size ${INFERENCE_BATCH_SIZE} --qlty_window_size ${QLTY_WINDOW_SIZE} --qlty_step_size ${QLTY_STEP_SIZE} #-save_results