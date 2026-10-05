#!/bin/bash
# ===========================
# Global set of parameters
# ===========================

# Tmon parameters
omega_p=1

# Zeta range (ZETA_NUM values, inclusive of both ends)
ZETA_START=0.40 # 0.05
ZETA_END=0.42
ZETA_NUM=1

# Coupling to readout resonator
# G=0.215      # GHz
# OMEGA_R=7.05 # GHz

# Hilbert space sizes
CUTOFF=41       # basis states for initial diagonalization
TMON_DIM=30     # Transmon eigenstates to keep for Floquet sim
# RESONATOR_DIM=5 # Readout resonator dimension

# Drive frequency range: NUM_FREQS uniform points (spacing 8.8 / 22000 = 0.0004),
# solved in consecutive batches of FREQS_PER_BATCH; the last batch also takes the
# remainder (21 x 1000 + 1 x 1001).
FREQ_RATIO_START=1.2
FREQ_RATIO_END=10.0
NUM_FREQS=22001
FREQS_PER_BATCH=1000

# Drive amplitude range
XI_SQ_START=0.0
XI_SQ_END=3.0 # XI^2 = 2 => CHI_AC = EC = 0.2 GHz (we want chi * nbar = 20e-3 * 15 = 0.3 GHz)
XI_SQ_NUM=201

# Gate charge range
# Note: the array in tmon_batched.sh must have ZETA_NUM * NG_NUM tasks
NG_START=0.0
NG_END=0.5
NG_NUM=26

# Hyperparameters for floquet
FIT_RANGE_FRACTION=0.5
FIT_CUTOFF=6
OVLP_CUTOFF=0.8
MAX_STEPS=30000
KEEP_FLOQUET_MODES=0

# Results (project NFS space; written only at the end of each task)
SAVE_ROOT=/home/hkb7/project_pi_sp979/hkb7/dust/tmon_gpu


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