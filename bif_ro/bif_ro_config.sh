#!/bin/bash
# ===========================
# Run settings for bif_ro_batched.sh
# The physics parameters (system, drive frequencies and photon numbers) are in params.py
# ===========================

# FREQS_PER_BATCH and AMP_CHUNK can be overridden from the environment, e.g.
#   sbatch --export=ALL,FREQS_PER_BATCH=10 bif_ro_batched.sh

# Drive frequencies solved per batch. The batch's Floquet modes stay on the GPU:
# FREQS_PER_BATCH x num_photon_nums x hilbert_dim^2 complex values, ~1.2 GB per
# frequency for hilbert_dim = 600 and 201 photon numbers, and twice that at the peak
# of the solve. The remaining frequencies are added to the last batch.
FREQS_PER_BATCH=${FREQS_PER_BATCH:-4}

# Drive amplitudes solved at once on the GPU (each call: FREQS_PER_BATCH x AMP_CHUNK points)
AMP_CHUNK=${AMP_CHUNK:-4}

# Hyperparameters for floquet
FIT_RANGE_FRACTION=1.0
FIT_CUTOFF=8
OVLP_CUTOFF=0.8
MAX_STEPS=30000
FIT_FREQ_CHUNK=100 # drive frequencies per chunk in the online refit over the full range
KEEP_FLOQUET_MODES=0

# Results (project NFS space; written only at the end of the job)
SAVE_ROOT=/home/hkb7/project_pi_sp979/hkb7/dust/bif_ro_gpu
