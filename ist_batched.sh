#!/bin/bash
#SBATCH --job-name=ist_flq_gpu
#SBATCH --partition=gpu_b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-204
#SBATCH -o out/%A-output-%a.txt -e out/%A-errors-%a.txt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=harsh.babla@yale.edu

set -euo pipefail
source ist_config.sh

# Sweep values, and this task's position in the (zeta, alpha_tilde) grid
mapfile -t ZETA_VALS < <(linspace "$ZETA_START" "$ZETA_END" "$ZETA_NUM")
mapfile -t ALPHA_TILDE_VALS < <(linspace "$ALPHA_TILDE_START" "$ALPHA_TILDE_END" "$ALPHA_TILDE_NUM")
ZETA_IDX=$(( SLURM_ARRAY_TASK_ID / ALPHA_TILDE_NUM ))
ALPHA_TILDE_IDX=$(( SLURM_ARRAY_TASK_ID % ALPHA_TILDE_NUM ))
if [ "$ZETA_IDX" -ge "$ZETA_NUM" ]; then
  echo "Error: array index $SLURM_ARRAY_TASK_ID exceeds ZETA_NUM * ALPHA_TILDE_NUM - 1" >&2
  exit 1
fi
ZETA=$(printf "%.4f" "${ZETA_VALS[$ZETA_IDX]}")
ALPHA_TILDE=$(printf "%.4f" "${ALPHA_TILDE_VALS[$ALPHA_TILDE_IDX]}")
echo "task $SLURM_ARRAY_TASK_ID: zeta[$ZETA_IDX] = $ZETA, alpha_tilde[$ALPHA_TILDE_IDX] = $ALPHA_TILDE"

# Compute in node-local /tmp (private to this job, wiped when it ends); copy the
# result to the shared filesystem only at the end.
WORK_DIR=/tmp/floquet_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
SAVE_DIR=$SAVE_ROOT/zeta_$ZETA
OUTPUT_NAME=zeta_${ZETA}_alpha_tilde_${ALPHA_TILDE}.h5
mkdir -p "$WORK_DIR" "$SAVE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load uv
uv run python ist_batched.py \
  --zeta="$ZETA" \
  --alpha_tilde="$ALPHA_TILDE" \
  --flux="$FLUX_BIAS" \
  --omega_0="$omega_0" \
  --cutoff="$CUTOFF" \
  --hilbert_dim="$HILBERT_DIM" \
  --state_indices="$STATE_INDICES" \
  --freq_ratio_start="$FREQ_RATIO_START" \
  --freq_ratio_end="$FREQ_RATIO_END" \
  --freq_ratio_num="$NUM_FREQS" \
  --freq_ratio_num_per_batch="$FREQS_PER_BATCH" \
  --xi_sq_start="$XI_SQ_START" \
  --xi_sq_end="$XI_SQ_END" \
  --xi_sq_num="$XI_SQ_NUM" \
  --fit_range_fraction="$FIT_RANGE_FRACTION" \
  --fit_cutoff="$FIT_CUTOFF" \
  --overlap_cutoff="$OVLP_CUTOFF" \
  --max_steps="$MAX_STEPS" \
  --keep_floquet_modes="$KEEP_FLOQUET_MODES" \
  --work_dir="$WORK_DIR" \
  --output_name="$OUTPUT_NAME"

# Copy under a temporary name, then rename, so a partial copy never looks complete.
cp "$WORK_DIR/$OUTPUT_NAME" "$SAVE_DIR/.$OUTPUT_NAME.partial"
mv "$SAVE_DIR/.$OUTPUT_NAME.partial" "$SAVE_DIR/$OUTPUT_NAME"
echo "saved $SAVE_DIR/$OUTPUT_NAME"
