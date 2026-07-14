#!/bin/bash

# Caution!!!!! This script written by Claude.
# Don't blame Harsh if your computer blows up (please)
#
# Submits validate_ozaki.py (ozaki_matmul / ozaki_solve accuracy+timing check,
# see ozaki.py) once per device, mirroring submit.sh's partition/GPU
# conventions. This is a standalone correctness+timing check, decoupled from
# the full solver benchmark pipeline -- not integrated into submit.sh.
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DEVICE_NAMES=(     'cpu'   'gpu_h200'        'gpu_rtx6000'                       'gpu_b200'     )
DEVICE_PARTITIONS=('day'   'gpu_h200'        'gpu_rtx6000'                       'gpu_b200'     )
DEVICE_GPU_FLAGS=( ''      '--gpus=h200:1'   '--gpus=rtx_pro_6000_blackwell:1'   '--gpus=b200:1')

JOB_TIME='00:20:00'
JOB_MEM_PER_CPU='10G'
MAIL_USER='harsh.babla@yale.edu'
WORK_DIR="$(pwd)"
OUT_DIR='out/ozaki_validate'

export WORK_DIR
mkdir -p "${OUT_DIR}"

# ──────────────────────────────────────────────────────────────────────────────
# submit_validate DEVICE PARTITION GPU_FLAG
#   Runs validate_ozaki.py once on one device. Prints the submitted job ID.
# ──────────────────────────────────────────────────────────────────────────────
submit_validate() {
    local device=$1 partition=$2 gpu_flag=$3

    [[ "${device:0:3}" == "gpu" ]] && local jax_plat="cuda,cpu" || local jax_plat="cpu"
    export DEVICE="${device}" JAX_PLAT="${jax_plat}"

    local args=(
        --parsable
        --partition="${partition}"
        --job-name="ozaki_validate_${device}"
        --ntasks=1 --nodes=1 --cpus-per-task=1
        --mem-per-cpu="${JOB_MEM_PER_CPU}"
        --time="${JOB_TIME}"
        --mail-type=BEGIN,END,FAIL
        --mail-user="${MAIL_USER}"
        -o "${OUT_DIR}/${device}_out.txt"
        -e "${OUT_DIR}/${device}_err.txt"
    )
    [[ -n "${gpu_flag}" ]] && args+=("${gpu_flag}")

    sbatch "${args[@]}" << 'BATCH'
#!/bin/bash
echo "Running validate_ozaki.py on device=${DEVICE} (JAX_PLATFORMS=${JAX_PLAT})"
cd "${WORK_DIR}" && module load uv
JAX_PLATFORMS=${JAX_PLAT} uv run python validate_ozaki.py
BATCH
}

# ──────────────────────────────────────────────────────────────────────────────
# Submit one job per device.
# ──────────────────────────────────────────────────────────────────────────────
echo "=== Ozaki matmul/solve validation: one job per device ==="
for i in "${!DEVICE_NAMES[@]}"; do
    DEVICE="${DEVICE_NAMES[$i]}"
    PARTITION="${DEVICE_PARTITIONS[$i]}"
    GPU_FLAG="${DEVICE_GPU_FLAGS[$i]}"

    jid=$(submit_validate "${DEVICE}" "${PARTITION}" "${GPU_FLAG}")
    echo "  ${DEVICE} (partition=${PARTITION}) -> job ${jid}, output: ${OUT_DIR}/${DEVICE}_out.txt"
done
