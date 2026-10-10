#!/bin/bash
# ===========================
# Global set of parameters
# ===========================

# IST parameters
omega_0=1.0
alpha_tilde=0.030
zeta=0.30 # default

# Zeta range (ZETA_NUM values, inclusive of both ends)
ZETA_START=0.10 # 0.05
ZETA_END=0.50
ZETA_NUM=41

# ALPHA_TILDE range (ALPHA_TILDE_NUM values, inclusive of both ends)
ALPHA_TILDE_START=0.010
ALPHA_TILDE_END=0.026
ALPHA_TILDE_NUM=5

# Hilbert space sizes
CUTOFF=200      # basis states for initial diagonalization
HILBERT_DIM=30  # IST eigenstates to keep for Floquet sim
STATE_INDICES="0,1" # IST eigenstates to perform the displaced state fitting to

# Drive frequency range: NUM_FREQS uniform points 
# solved in consecutive batches of FREQS_PER_BATCH.
FREQ_RATIO_START=1.4
FREQ_RATIO_END=10.2
NUM_FREQS=22001
FREQS_PER_BATCH=500

# Drive amplitude range
XI_SQ_START=0.0
XI_SQ_END=5.0
XI_SQ_NUM=251

# Flux bias
FLUX_BIAS=0.5

# Hyperparameters for floquet
FIT_RANGE_FRACTION=0.1
FIT_CUTOFF=6
OVLP_CUTOFF=0.90
MAX_STEPS=30000
KEEP_FLOQUET_MODES=0

# Results (project NFS space; written only at the end of each task)
SAVE_ROOT=/home/hkb7/project_pi_sp979/hkb7/dust/ist_gpu2


# ===========================
# Linspace function
# ===========================
linspace() {
  start=$1
  stop=$2
  num=$3

  if [ "$num" -lt 1 ]; then
    echo "Error: num must be greater than 0" >&2
    return 1

  elif [ "$num" -eq 1 ]; then
    echo $start
    return 0

  elif [ "$num" -eq 2 ]; then
    echo $start
    echo $stop
    return 0
  fi

  step=$(echo "scale=10; ($stop - $start) / ($num - 1)" | bc)

  for (( i=0; i<num; i++ )); do
    value=$(echo "scale=10; $start + $i * $step" | bc)
    echo $value
  done
}