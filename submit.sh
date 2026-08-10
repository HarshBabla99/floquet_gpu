#!/bin/bash

# Caution!!!!! This script written by Claude.
# Don't blame Harsh if your computer blows up (please)
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_JOBS_PER_SOLVER=1
DIMS=(2 4 8 16 32 64 128 256 512 1024 2048 4096)
# DIMS=(8192 16384 32768)
SOLVERS=('dq_basic' 'cayley')

# Devices: name  partition  gpu_flag ('' for CPU)
DEVICE_NAMES=(     'cpu'   'gpu_h200'     )
DEVICE_PARTITIONS=('day'   'gpu_h200'     )
DEVICE_GPU_FLAGS=( ''      '--gpus=h200:1')
# DEVICE_NAMES=(     'gpu_h200'        'gpu_b200'     )
# DEVICE_PARTITIONS=('gpu_h200'        'gpu_b200'     )
# DEVICE_GPU_FLAGS=( '--gpus=h200:1'   '--gpus=b200:1')

# ── Run variants ──────────────────────────────────────────────────────────────
# Each entry is "tag|VAR=value;VAR=value;...". The assignments are exported into
# the benchmark process, so a variant can set XLA_FLAGS, library env vars
# (cuBLAS emulation, etc.), or any combination. Values may contain spaces;
# separate assignments with ';'. Every (device x variant x solver x dim x run)
# combination is submitted.
#
# The tag labels the output tree (out/{device}_{tag}/, consolidated to
# out/{device}_{tag}.npy) and is recorded in every row, together with a snapshot
# of the XLA_*/CUBLAS_*/JAX_*/CUDA_* environment actually in effect. So variants
# can never be mixed in one file, and a merged analysis can still group by them.
# Use an empty tag for a single untagged run.
#
# NOTE: device-time attribution is NOT comparable across variants. With
# +WHILE command buffers the loop's kernels run inside a CUDA graph and the
# profiler reports graph launch cost rather than kernel cost, so `device_busy_ns`
# collapses (~7x at d=4096) while actual runtime is unchanged. Compare `t_total`
# across variants; compare `device_busy_ns` only within a variant.
#
# Job count is NUM_DIMS x NUM_JOBS_PER_SOLVER x #SOLVERS x #VARIANTS per device.
GPU_VARIANTS=(
    # Let XLA capture Tsit5's adaptive-step while-loop as one CUDA graph.
    # Requires CUDA >= 12.3.
    #"cmdbuf|XLA_FLAGS=--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
    # XLA's default command-buffer set (FUSION, CUBLAS, ... but not WHILE).
    # Keep this one: it is the only variant with faithful per-kernel attribution,
    # so profiling.py's propagator breakdown must be built from its traces.
    "nocmdbuf|"
    # Ozaki-I scheme: cuBLAS emulates FP64 GEMM with lower-precision tensor-core
    # products instead of using native FP64 units. 'eager' applies it wherever
    # cuBLAS can rather than only where it heuristically expects a win.
    #
    # THREE THINGS TO CHECK BEFORE TRUSTING THIS VARIANT - see README step 8:
    #  1. That it is active at all. If the installed cuBLAS predates the feature
    #     the env vars are ignored and you get a relabelled duplicate of nocmdbuf.
    #  2. That it applies to ZGEMM. This workload is complex128 throughout; the
    #     emulation is documented mainly for real DGEMM. If it only covers real
    #     matmuls, this variant is a no-op here.
    #  3. That accuracy still holds. Emulated FP64 is not bit-identical, and
    #     Tsit5 is ADAPTIVE at rtol=atol=1e-8 - degraded GEMM accuracy changes the
    #     number of steps taken, so a t_total difference would conflate 'faster
    #     matmul' with 'different amount of work'.
    # "ozaki|CUBLAS_EMULATE_DOUBLE_PRECISION=1;CUBLAS_EMULATION_STRATEGY=eager;XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION"
    #"all_cmdbuf_off|XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION"

)

# The variants above are GPU-specific; CPU gets a single untagged run by default.
# Add entries here to sweep CPU-side settings the same way.
CPU_VARIANTS=(
    "|"
)

# Each job now runs the solver three times (warmup, timed, traced) rather than
# twice, since the timed run is no longer the profiled one.
BENCH_TIME='01:30:00'
BENCH_MEM_PER_CPU='32G'
MAIL_USER='harsh.babla@yale.edu'
WORK_DIR="$(pwd)"

# Derived; exported so batch scripts receive them via SLURM's --export=ALL default
NUM_DIMS=${#DIMS[@]}
export NUM_JOBS_PER_SOLVER NUM_DIMS WORK_DIR
export DIMS_STR="${DIMS[*]}"

# ──────────────────────────────────────────────────────────────────────────────
# submit_solver SOLVER DEVICE PARTITION GPU_FLAG DEP_STR
#   Submits a NUM_DIMS × NUM_JOBS_PER_SOLVER array for one solver on one device.
#   DEP_STR: colon-separated SLURM job IDs, or '' for no dependency.
#   Prints the submitted job ID to stdout.
# ──────────────────────────────────────────────────────────────────────────────
#   Reads TAG, VARIANT_ENV and OUT_SUBDIR from the enclosing device loop.
submit_solver() {
    local solver=$1 device=$2 partition=$3 gpu_flag=$4 dep_str=$5

    # Bake solver/device into the environment for the batch script
    [[ "${device:0:3}" == "gpu" ]] && local jax_plat="cuda,cpu" || local jax_plat="cpu"

    export SOLVER="${solver}" DEVICE="${device}" JAX_PLAT="${jax_plat}" \
           VARIANT_ENV="${VARIANT_ENV}" TAG="${TAG}"

    local args=(
        --parsable
        --array="0-$(( NUM_DIMS * NUM_JOBS_PER_SOLVER - 1 ))"
        --partition="${partition}"
        --job-name="${solver}_${OUT_SUBDIR}"
        --ntasks=1 --nodes=1 --cpus-per-task=1
        --mem-per-cpu="${BENCH_MEM_PER_CPU}"
        --time="${BENCH_TIME}"
        --mail-type=BEGIN,END,FAIL
        --mail-user="${MAIL_USER}"
        -o "out/${OUT_SUBDIR}/${solver}_out_%a.txt"
        -e "out/${OUT_SUBDIR}/${solver}_err_%a.txt"
    )
    [[ -n "${dep_str}"  ]] && args+=(--dependency="afterany:${dep_str}")
    [[ -n "${gpu_flag}" ]] && args+=("${gpu_flag}")

    sbatch "${args[@]}" << 'BATCH'
#!/bin/bash
read -ra DIMS <<< "${DIMS_STR}"
DIM_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_JOBS_PER_SOLVER ))
RUN_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_JOBS_PER_SOLVER ))
DIM=${DIMS[$DIM_IDX]}
echo "Task ${SLURM_ARRAY_TASK_ID}: ${SOLVER} d=${DIM} r=${RUN_IDX} on ${DEVICE} [${TAG:-none}]"
cd "${WORK_DIR}" && module load uv

# Apply this variant's environment assignments. Each is a whole VAR=value word,
# so values containing spaces (e.g. several XLA flags) survive intact.
if [[ -n "${VARIANT_ENV}" ]]; then
    IFS=';' read -ra _assignments <<< "${VARIANT_ENV}"
    for _a in "${_assignments[@]}"; do
        [[ -n "${_a}" ]] && export "${_a}"
    done
fi
echo "  variant env: ${VARIANT_ENV:-none}"

JAX_PLATFORMS="${JAX_PLAT}" uv run python benchmark.py \
    --solver "${SOLVER}" --dim "${DIM}" --run-index "${RUN_IDX}" \
    --device "${DEVICE}" --tag "${TAG}"
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
        --partition="scavenge" \
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
# Submit solvers, one device at a time.
# Within a device all solver arrays run in parallel; consolidation waits for all.
# ──────────────────────────────────────────────────────────────────────────────
for i in "${!DEVICE_NAMES[@]}"; do
    DEVICE="${DEVICE_NAMES[$i]}"
    PARTITION="${DEVICE_PARTITIONS[$i]}"
    GPU_FLAG="${DEVICE_GPU_FLAGS[$i]}"

    # Pick the variant list for this device class.
    if [[ "${DEVICE:0:3}" == "gpu" ]]; then
        VARIANTS=("${GPU_VARIANTS[@]}")
    else
        VARIANTS=("${CPU_VARIANTS[@]}")
    fi

    echo ""
    echo "=== ${i}: ${DEVICE} (partition=${PARTITION}, ${#VARIANTS[@]} variant(s)) ==="

    for variant in "${VARIANTS[@]}"; do
        # Split "tag|env" on the first '|'; either side may be empty.
        TAG="${variant%%|*}"
        VARIANT_ENV="${variant#*|}"
        OUT_SUBDIR="${DEVICE}${TAG:+_${TAG}}"
        export TAG VARIANT_ENV OUT_SUBDIR

        echo "  -- variant tag=${TAG:-none} env=${VARIANT_ENV:-none} → out/${OUT_SUBDIR}/"
        mkdir -p "out/${OUT_SUBDIR}"

        SOLVER_JOB_IDS=()
        for solver in "${SOLVERS[@]}"; do
            jid=$(submit_solver "${solver}" "${DEVICE}" "${PARTITION}" "${GPU_FLAG}" "")
            SOLVER_JOB_IDS+=("${jid}")
            echo "     ${solver} → ${jid}"
        done

        SOLVER_DEP=$(IFS=':'; echo "${SOLVER_JOB_IDS[*]}")

        # Consolidate after all solver arrays for THIS variant finish
        # (afterany = regardless of exit status)
        CONSOLIDATE_JOB=$(submit_consolidate "${OUT_SUBDIR}" "${WORK_DIR}/out/${OUT_SUBDIR}" \
            "${WORK_DIR}/out/${OUT_SUBDIR}.npy" "${SOLVERS[*]}" "${SOLVER_DEP}")
        echo "     Consolidate → ${CONSOLIDATE_JOB}"
    done
done
