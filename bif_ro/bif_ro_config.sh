#!/bin/bash
# ===========================
# Run settings for bif_ro_batched.sh
# The physics parameters (system, drive frequencies and photon numbers) are in params.py
# ===========================

# Drive frequencies solved per batch. Each batch holds
# FREQS_PER_BATCH x num_photon_nums x hilbert_dim^2 complex Floquet modes on the host,
# i.e. ~1.2 GB per frequency for hilbert_dim = 600 and 201 photon numbers.
FREQS_PER_BATCH=4

# Drive amplitudes solved at once on the GPU (each call: FREQS_PER_BATCH x AMP_CHUNK points)
AMP_CHUNK=8

# Hyperparameters for floquet
FIT_RANGE_FRACTION=1.0
FIT_CUTOFF=8
OVLP_CUTOFF=0.8
MAX_STEPS=30000
FIT_FREQ_CHUNK=100 # drive frequencies per chunk in the online refit over the full range
KEEP_FLOQUET_MODES=0

# Results (project NFS space; written only at the end of the job)
SAVE_ROOT=/home/hkb7/project_pi_sp979/hkb7/dust/bif_ro_gpu
