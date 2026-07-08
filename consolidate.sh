#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=consolidate_gpu_h200
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=00:05:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=harsh.babla@yale.edu
#SBATCH -o out/gpu_h200/_consolidate_out.txt
#SBATCH -e out/gpu_h200/_consolidate_err.txt

# Edit RAW_DIR/OUTPUT and the #SBATCH partition/output lines, to match the device
set -euo pipefail

RAW_DIR='out/gpu_h200'
OUTPUT='out/gpu_h200.npy'

# Either the list below or just 'basic'
SOLVERS=('dq_basic' 'cayley' 'sambe_sparse' 'sambe_dense'
         'dq_basic_jit' 'cayley_jit' 'sambe_sparse_jit' 'sambe_dense_jit')

module load uv

uv run python consolidate.py \
    --out-dir "${RAW_DIR}/" \
    --output "${OUTPUT}" \
    --solvers "${SOLVERS[@]}"
