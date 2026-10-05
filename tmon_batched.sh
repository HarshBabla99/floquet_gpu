#!/bin/bash
#SBATCH --job-name=tmon_flq_gpu
#SBATCH --partition=gpu_b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-77%15
#SBATCH -o out/%A-output-%a.txt -e out/%A-errors-%a.txt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=harsh.babla@yale.edu
#
# One array task per (zeta, ng) pair: submit with `sbatch tmon_batched.sh`.
# The array must have ZETA_NUM * NG_NUM tasks (0-77 for 3 x 26). 
# A 1000 x 201 batch at TMON_DIM=30 needs ~62 GB of GPU memory, so use
# >= 80 GB GPUs (b200 / h200 / rtx_pro_6000_blackwell).

set -euo pipefail
source config.sh

# Sweep values, and this task's position in the (zeta, ng) grid
mapfile -t ZETA_VALS < <(linspace "$ZETA_START" "$ZETA_END" "$ZETA_NUM")
mapfile -t NG_VALS < <(linspace "$NG_START" "$NG_END" "$NG_NUM")
ZETA_IDX=$(( SLURM_ARRAY_TASK_ID / NG_NUM ))
NG_IDX=$(( SLURM_ARRAY_TASK_ID % NG_NUM ))
if [ "$ZETA_IDX" -ge "$ZETA_NUM" ]; then
  echo "Error: array index $SLURM_ARRAY_TASK_ID exceeds ZETA_NUM * NG_NUM - 1" >&2
  exit 1
fi
ZETA=$(printf "%.4f" "${ZETA_VALS[$ZETA_IDX]}")
NG=$(printf "%.4f" "${NG_VALS[$NG_IDX]}")
echo "task $SLURM_ARRAY_TASK_ID: zeta[$ZETA_IDX] = $ZETA, ng[$NG_IDX] = $NG"

# Compute in node-local /tmp (private to this job, wiped when it ends); copy the
# result to the shared filesystem only at the end.
WORK_DIR=/tmp/floquet_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
SAVE_DIR=$SAVE_ROOT/zeta_$ZETA
OUTPUT_NAME=zeta_${ZETA}_ng_${NG}.h5
mkdir -p "$WORK_DIR" "$SAVE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load uv
uv run python tmon_batched.py \
  --zeta="$ZETA" \
  --ng="$NG" \
  --omega_p="$omega_p" \
  --ncut="$CUTOFF" \
  --hilbert_dim="$TMON_DIM" \
  --freq_ratio_start="$FREQ_RATIO_START" \
  --freq_ratio_end="$FREQ_RATIO_END" \
  --num_freqs="$NUM_FREQS" \
  --freqs_per_batch="$FREQS_PER_BATCH" \
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
