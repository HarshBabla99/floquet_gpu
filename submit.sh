#!/bin/bash

# Caution!!!!! This script written by Claude.
# Don't blame Harsh if your computer blows up (please)
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_JOBS_PER_SOLVER=5
DIMS=(2 4 8 16 32 64 128 256 512 1024 2048 4096)
NON_BASIC_SOLVERS=('dq_basic' 'cayley' 'sambe_sparse' 'sambe_dense'
                   'dq_basic_jit' 'cayley_jit' 'sambe_sparse_jit' 'sambe_dense_jit')

BASIC_PARTITION='day'
BASIC_DIR='out/cpu'

# Non-basic devices: name  partition  gpu_flag ('' for CPU)
DEVICE_NAMES=(     'cpu'   'gpu_h200'        'gpu_rtx6000'                       'gpu_b200'     )
DEVICE_PARTITIONS=('day'   'gpu_h200'        'gpu_rtx6000'                       'gpu_b200'     )
DEVICE_GPU_FLAGS=( ''      '--gpus=h200:1'   '--gpus=rtx_pro_6000_blackwell:1'   '--gpus=b200:1')

BENCH_TIME='00:30:00'
BENCH_MEM_PER_CPU='10G'
MAIL_USER='harsh.babla@yale.edu'
WORK_DIR="$(pwd)"

# Derived; exported so batch scripts receive them via SLURM's --export=ALL default
NUM_DIMS=${#DIMS[@]}
export NUM_JOBS_PER_SOLVER NUM_DIMS BASIC_DIR WORK_DIR
export DIMS_STR="${DIMS[*]}"

# ──────────────────────────────────────────────────────────────────────────────
# submit_solver SOLVER DEVICE PARTITION GPU_FLAG DEP_STR
#   Submits a NUM_DIMS × NUM_JOBS_PER_SOLVER array for one solver on one device.
#   DEP_STR: colon-separated SLURM job IDs, or '' for no dependency.
#   Prints the submitted job ID to stdout.
# ──────────────────────────────────────────────────────────────────────────────
submit_solver() {
    local solver=$1 device=$2 partition=$3 gpu_flag=$4 dep_str=$5

    # Bake solver/device into the environment for the batch script
    [[ "${device:0:3}" == "gpu" ]] && local jax_plat="cuda,cpu" || local jax_plat="cpu"
    export SOLVER="${solver}" DEVICE="${device}" JAX_PLAT="${jax_plat}"

    local args=(
        --parsable
        --array="0-$(( NUM_DIMS * NUM_JOBS_PER_SOLVER - 1 ))"
        --partition="${partition}"
        --job-name="${solver}_${device}"
        --ntasks=1 --nodes=1 --cpus-per-task=1
        --mem-per-cpu="${BENCH_MEM_PER_CPU}"
        --time="${BENCH_TIME}"
        --mail-type=BEGIN,END,FAIL
        --mail-user="${MAIL_USER}"
        -o "out/${device}/${solver}_out_%a.txt"
        -e "out/${device}/${solver}_err_%a.txt"
    )
    [[ -n "${dep_str}"  ]] && args+=(--dependency="afterany:${dep_str}")
    [[ -n "${gpu_flag}" ]] && args+=("${gpu_flag}")

    sbatch "${args[@]}" << 'BATCH'
#!/bin/bash
read -ra DIMS <<< "${DIMS_STR}"
DIM_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_JOBS_PER_SOLVER ))
RUN_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_JOBS_PER_SOLVER ))
DIM=${DIMS[$DIM_IDX]}
echo "Task ${SLURM_ARRAY_TASK_ID}: ${SOLVER} d=${DIM} r=${RUN_IDX} on ${DEVICE}"
cd "${WORK_DIR}" && module load uv
JAX_PLATFORMS=${JAX_PLAT} uv run python benchmark.py \
    --solver "${SOLVER}" --dim "${DIM}" --run-index "${RUN_IDX}" \
    --device "${DEVICE}" --basic-dir "${BASIC_DIR}"
BATCH
}

# ──────────────────────────────────────────────────────────────────────────────
# submit_consolidate NAME RAW_DIR OUTPUT SOLVERS_STR DEP_STR
#   Submits the consolidation job that aggregates one or more solvers' raw
#   per-task outputs (in RAW_DIR) into a single OUTPUT .npy file, after DEP_STR
#   (colon-separated SLURM job IDs) finishes.
#   Prints the submitted job ID to stdout.
# ──────────────────────────────────────────────────────────────────────────────
submit_consolidate() {
    local name=$1 raw_dir=$2 output=$3 solvers_str=$4 dep_str=$5

    sbatch \
        --parsable \
        --partition="${BASIC_PARTITION}" \
        --job-name="consolidate_${name}" \
        --ntasks=1 --mem=5G --time=00:05:00 \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="${MAIL_USER}" \
        -o "${raw_dir}/_consolidate_${name}_out.txt" \
        -e "${raw_dir}/_consolidate_${name}_err.txt" \
        --dependency="afterany:${dep_str}" \
        --wrap="cd ${WORK_DIR} && module load uv && \
uv run python consolidate.py \
    --out-dir ${raw_dir}/ \
    --output ${output} \
    --solvers ${solvers_str}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: basic (qutip, always CPU)
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p "${BASIC_DIR}"
echo "=== Phase 1: basic ==="
BASIC_JOB=$(submit_solver 'basic' 'cpu' "${BASIC_PARTITION}" '' '')
echo "  Job array: ${BASIC_JOB}"

# Consolidate after the basic job finishes
CONSOLIDATE_JOB=$(submit_consolidate 'basic' "${WORK_DIR}/${BASIC_DIR}" \
    "${WORK_DIR}/out/basic.npy" 'basic' "${BASIC_JOB}")
echo "  Consolidate → ${CONSOLIDATE_JOB}"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: non-basic solvers, one device at a time.
# Within a device all solver arrays run in parallel; consolidation waits for all.
# ──────────────────────────────────────────────────────────────────────────────
for i in "${!DEVICE_NAMES[@]}"; do
    DEVICE="${DEVICE_NAMES[$i]}"
    PARTITION="${DEVICE_PARTITIONS[$i]}"
    GPU_FLAG="${DEVICE_GPU_FLAGS[$i]}"

    echo ""
    echo "=== Phase 2.${i}: non-basic on ${DEVICE} (partition=${PARTITION}) ==="
    mkdir -p "out/${DEVICE}"

    SOLVER_JOB_IDS=()
    for solver in "${NON_BASIC_SOLVERS[@]}"; do
        jid=$(submit_solver "${solver}" "${DEVICE}" "${PARTITION}" "${GPU_FLAG}" "${BASIC_JOB}")
        SOLVER_JOB_IDS+=("${jid}")
        echo "  ${solver} → ${jid}"
    done

    SOLVER_DEP=$(IFS=':'; echo "${SOLVER_JOB_IDS[*]}")

    # Consolidate after all solver arrays finish (afterany = regardless of exit status)
    CONSOLIDATE_JOB=$(submit_consolidate "${DEVICE}" "${WORK_DIR}/out/${DEVICE}" \
        "${WORK_DIR}/out/${DEVICE}.npy" "${NON_BASIC_SOLVERS[*]}" "${SOLVER_DEP}")
    echo "  Consolidate → ${CONSOLIDATE_JOB}"
done
