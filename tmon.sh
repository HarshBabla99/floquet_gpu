#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=tmon_flq
#SBATCH -o out/%j-output-%a.txt -e out/%j-errors-%a.txt
#SBATCH --array=0-25
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5G
#SBATCH --time=05:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=harsh.babla@yale.edu

######################
# Import config
source config.sh

# These are set by run_all.sh via --export. Fail loudly rather than silently
# falling back to some other zeta / output location.
: "${ZETA:?ZETA not exported by run_all.sh}"
: "${SAVE_DIRECTORY:?SAVE_DIRECTORY not exported by run_all.sh}"

##################################

# Select the value of ng for this job
NG_ARRAY=()
while IFS= read -r line; do
  NG_ARRAY+=("$line")
done < <(linspace $NG_START $NG_END $NG_NUM)
ng=${NG_ARRAY[${SLURM_ARRAY_TASK_ID}]}
echo "$ng"

# Arrange the qubit parameters
QUBIT_PARAMS="{'omega_p': $omega_p, 'zeta': $ZETA, 'ng': $ng, 'ncut': $CUTOFF}"

# Drive frequencies (save into a temp file)
echo "$START"
echo "$END"
echo "$NUM"

FREQ_RATIO_VALS=$(linspace $START $END $NUM)
FREQS_TMP=$(mktemp)
echo "$FREQ_RATIO_VALS" > "$FREQS_TMP"

# Drive amplitudes (save into a temp file) 
echo "$XI_SQ_START"
echo "$XI_SQ_END"
echo "$XI_SQ_NUM"

XI_SQ_VALS=$(linspace $XI_SQ_START $XI_SQ_END $XI_SQ_NUM)
XI_SQ_TMP=$(mktemp)
echo "$XI_SQ_VALS" > "$XI_SQ_TMP"

# Output directory is constructed in run_all.sh and passed in via --export
echo "zeta = $ZETA"
echo "save directory = $SAVE_DIRECTORY"
mkdir -p "$SAVE_DIRECTORY"

############################

module load uv
uv run python tmon.py \
  --idx=$SLURM_ARRAY_TASK_ID \
  --hilbert_dim=$TMON_DIM \
  --qubit_params="$QUBIT_PARAMS" \
  --drive_freq_ratios_file="$FREQS_TMP" \
  --xi_sq_file="$XI_SQ_TMP" \
  --fit_range_fraction=$FIT_RANGE_FRACTION \
  --floquet_sampling_time_fraction=$FLOQUET_SAMPLING_TIME_FRACTION \
  --fit_cutoff=$FIT_CUTOFF \
  --overlap_cutoff=$OVLP_CUTOFF \
  --nsteps=$NSTEPS \
  --num_cpus="$SLURM_CPUS_PER_TASK" \
  --save_floquet_modes=$SAVE_FLOQUET_MODES \
  --save_directory=$SAVE_DIRECTORY 

rm "$FREQS_TMP"
rm "$XI_SQ_TMP"

mv out/${SLURM_JOB_ID}-output-${SLURM_ARRAY_TASK_ID}.txt ${SAVE_DIRECTORY}/output-${SLURM_ARRAY_TASK_ID}.txt
mv out/${SLURM_JOB_ID}-errors-${SLURM_ARRAY_TASK_ID}.txt ${SAVE_DIRECTORY}/errors-${SLURM_ARRAY_TASK_ID}.txt