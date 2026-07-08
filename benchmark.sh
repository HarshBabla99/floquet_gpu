#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=dq_basic_gpu_h200
#SBATCH --array=0-59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h200:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=harsh.babla@yale.edu
#SBATCH -o out/gpu_h200/dq_basic_out_%a.txt
#SBATCH -e out/gpu_h200/dq_basic_err_%a.txt

# Edit SOLVER, DEVICE, and the #SBATCH partition/gpu/output lines above
# Number of array jobs should equal NUM_DIMS * NUM_JOBS_PER_SOLVER
set -euo pipefail

# SOLVER choose from ('dq_basic' 'cayley' 'sambe_sparse' 'sambe_dense'
#                      'dq_basic_jit' 'cayley_jit' 'sambe_sparse_jit' 'sambe_dense_jit')
SOLVER="dq_basic"

# DEVICE name e.g. cpu, gpu_h200, gpu_rtx6000, gpu_b200
DEVICE="gpu_h200"

# JAX_PLAT = "cpu" for cpu, "cuda,cpu" for gpu
JAX_PLAT="cuda,cpu"

# Must match DIMS/NUM_JOBS_PER_SOLVER in submit.sh (and the --array range above:
# 0 .. NUM_DIMS*NUM_JOBS_PER_SOLVER-1)
DIMS=(2 4 8 16 32 64 128 256 512 1024 2048 4096)
NUM_JOBS_PER_SOLVER=5
BASIC_DIR='out/cpu'

DIM_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_JOBS_PER_SOLVER ))
RUN_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_JOBS_PER_SOLVER ))
DIM=${DIMS[$DIM_IDX]}
echo "Task ${SLURM_ARRAY_TASK_ID}: ${SOLVER} d=${DIM} r=${RUN_IDX} on ${DEVICE}"

module load uv

JAX_PLATFORMS=${JAX_PLAT} uv run python benchmark.py \
    --solver "${SOLVER}" --dim "${DIM}" --run-index "${RUN_IDX}" \
    --device "${DEVICE}" --basic-dir "${BASIC_DIR}"
