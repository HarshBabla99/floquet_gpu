#!/bin/bash
# ===========================
# Global set of parameters
# ===========================

# Tmon parameters
omega_p=1

# Zeta range (swept, one set of jobs + one merge per zeta value)
ZETA_START=0.38 # 0.05
ZETA_END=0.42
ZETA_STEP=0.02

# Seconds to wait between submitting one zeta's batch of jobs and the next, to stay
# under the cluster's max-submitted-jobs limit. Set to 0 to submit everything at once.
ZETA_SUBMIT_DELAY=100

# Coupling to readout resonator
# G=0.215      # GHz
# OMEGA_R=7.05 # GHz

# Hilbert space sizes
CUTOFF=41       # basis states for initial diagonalization
TMON_DIM=30     # Transmon eigenstates to keep for Floquet sim
# RESONATOR_DIM=5 # Readout resonator dimension

# Drive frequency range
FREQ_RATIO_START=1.2
FREQ_RATIO_END=10.0
FREQ_RATIO_STEP=0.2
NUM_FREQS_PER_BATCH=501

# Drive amplitude range
XI_SQ_START=0.0
XI_SQ_END=3.0 # XI^2 = 2 => CHI_AC = EC = 0.2 GHz (we want chi * nbar = 20e-3 * 15 = 0.3 GHz)
XI_SQ_NUM=201

# Gate charge range
# Note: Number of gate charges should match SLURM_ARRAY_TASK_COUNT
NG_START=0.0
NG_END=0.5
NG_NUM=26

# Hyperparameters for floquet
FIT_RANGE_FRACTION=0.5
FIT_CUTOFF=6
OVLP_CUTOFF=0.8
FLOQUET_SAMPLING_TIME_FRACTION=0.0
NSTEPS=30000
SAVE_FLOQUET_MODES=1


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