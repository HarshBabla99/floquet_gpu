#!/bin/bash
#SBATCH --job-name=bif_ro_flq_gpu
#SBATCH --partition=gpu_b200,gpu_h200,gpu_h100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o out/%j-output.txt -e out/%j-errors.txt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=harsh.babla@yale.edu
#
# Submit from the bif_ro directory: `sbatch bif_ro_batched.sh`.
# The extra CPUs speed up the (CPU) solve for the bare system parameters at every
# resonator frequency, which diagonalizes the coupled Hamiltonian many times.

set -euo pipefail
source bif_ro_config.sh

# Compute in node-local /tmp (private to this job, wiped when it ends); copy the
# result to the shared filesystem only at the end.
WORK_DIR=/tmp/floquet_${SLURM_JOB_ID}
SAVE_DIR=$SAVE_ROOT
OUTPUT_NAME=bif_ro_${SLURM_JOB_ID}.h5
mkdir -p "$WORK_DIR" "$SAVE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# The floquet package lives in the repo root (the parent of this directory) and is not
# installed into the environment, so make it importable from here.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
module load uv
uv run python bif_ro_batched.py \
  --freqs_per_batch="$FREQS_PER_BATCH" \
  --amp_chunk="$AMP_CHUNK" \
  --fit_range_fraction="$FIT_RANGE_FRACTION" \
  --fit_cutoff="$FIT_CUTOFF" \
  --overlap_cutoff="$OVLP_CUTOFF" \
  --max_steps="$MAX_STEPS" \
  --fit_freq_chunk="$FIT_FREQ_CHUNK" \
  --keep_floquet_modes="$KEEP_FLOQUET_MODES" \
  --work_dir="$WORK_DIR" \
  --output_name="$OUTPUT_NAME"

# Copy under a temporary name, then rename, so a partial copy never looks complete.
cp "$WORK_DIR/$OUTPUT_NAME" "$SAVE_DIR/.$OUTPUT_NAME.partial"
mv "$SAVE_DIR/.$OUTPUT_NAME.partial" "$SAVE_DIR/$OUTPUT_NAME"
echo "saved $SAVE_DIR/$OUTPUT_NAME"
