#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH -A <YOUR ACCOUNT>
#SBATCH --time=1-00:00:00
#SBATCH --qos=regular

#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

#SBATCH -e logs/%x-%j.err
#SBATCH -o logs/%x-%j.out

# Launch NVIDIA Dynamo (SGLang backend) serving Qwen3.6-27B at 1M context
# on Perlmutter. The server stays up for the wall-time so an external
# benchmark driver (bench_sglang_qwen3.6_1m.sh) can hit the OpenAI-compatible
# endpoint at http://${HEAD_NODE}:8000/v1.
#
# Submit:
#   sbatch launch_sglang_qwen3.6-1m.sh                 # 1 node, TP=4
#   sbatch --nodes=2 launch_sglang_qwen3.6-1m.sh       # 2 nodes, TP=8
#
# Override knobs at submit time, e.g.:
#   CONTEXT_LEN=524288 MAX_RUNNING=32 sbatch launch_sglang_qwen3.6-1m.sh

set -eE

export HF_HOME=$SCRATCH/cache
export TVM_FFI_CACHE_DIR=$SCRATCH/.cache
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null || true)}"
export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1"
export DYNAMO_STORE="$SCRATCH/dynamo_store_kv_${SLURM_JOB_ID}"
FRONTEND_LOG="logs/dynamo-frontend-${SLURM_JOB_ID}.log"
WORKER_LOG_PREFIX="logs/dynamo-worker-${SLURM_JOB_ID}"
READY_MARKER="logs/dynamo-ready-${SLURM_JOB_ID}.env"

mkdir -p logs "$SCRATCH"/root "$HF_HOME" "$TVM_FFI_CACHE_DIR" "$DYNAMO_STORE"
rm -f "${FRONTEND_LOG}" "${WORKER_LOG_PREFIX}"*.log "${READY_MARKER}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

podman_dynamo() {
    podman-hpc run \
    --rm \
    -u 0 \
    --gpu \
    --nccl-cu12 \
    --net host \
    --ipc=host \
    --ulimit memlock=-1 \
    -v "$SCRATCH"/root:/root \
    -v "$SCRATCH":"$SCRATCH" \
    -v "${DYNAMO_STORE}:/tmp/dynamo_store_kv" \
    -v "${PWD}:/workdir" \
    -e "NCCL_CUMEM_HOST_ENABLE=0" \
    -e "RUST_LOG=warn" \
    -e "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
    -e "TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR}" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    -e "HF_HOME=${HF_HOME}" \
    -w /tmp \
    ${DYNAMO_IMAGE} \
    "$@"
}
export -f podman_dynamo

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALL_NODES="${NODES[*]}"
HEAD_NODE="${NODES[0]}"

export LLM_HOST=0.0.0.0
export LLM_PORT=8000

# ── Model + long-context knobs ────────────────────────────────────────────────
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export TP="${TP:-$(( 4 * SLURM_JOB_NUM_NODES ))}"

# 1M context by default; lower if KV cache won't fit.
export CONTEXT_LEN="${CONTEXT_LEN:-1010000}"

# KV cache dtype — fp8 is required to fit 1M ctx for a 27B GQA model.
# Set KV_DTYPE=auto to disable fp8 KV (bf16) for a non-1M run.
export KV_DTYPE="${KV_DTYPE:-fp8_e5m2}"

# Prefill: chunked-prefill chunk size and per-step prefill token budget.
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"

# Decode: cap on concurrent in-flight requests.
export MAX_RUNNING="${MAX_RUNNING:-16}"

# Memory fraction reserved by SGLang for weights+KV (rest is activations).
export MEM_FRAC="${MEM_FRAC:-0.90}"

# Attention backend: flashinfer works on A100/H100. Fallback: triton.
export ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"

# Optional speculative decoding (MTP-equivalent). Leave SPEC_ALGO empty to
# disable. If enabled, you must also supply SPEC_DRAFT.
export SPEC_ALGO="${SPEC_ALGO:-}"
export SPEC_STEPS="${SPEC_STEPS:-2}"
export SPEC_DRAFT="${SPEC_DRAFT:-}"

# RoPE scaling override. Qwen3.6-27B ships with max_position_embeddings=262144
# (256k). Running past that REQUIRES YaRN scaling for valid outputs; just
# enlarging --context-length without rope scaling produces incoherent text.
# factor = CONTEXT_LEN / original_max_position_embeddings, e.g. 1010000/262144 ≈ 3.85
# Set ROPE_OVERRIDE to an empty string only if you keep CONTEXT_LEN <= 262144.
NATIVE_CTX=262144
if [ "${CONTEXT_LEN}" -gt "${NATIVE_CTX}" ]; then
    ROPE_FACTOR=$(python3 -c "print(round(${CONTEXT_LEN}/${NATIVE_CTX}+0.05, 2))")
    DEFAULT_ROPE_OVERRIDE="{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":${ROPE_FACTOR},\"original_max_position_embeddings\":${NATIVE_CTX}}}"
else
    DEFAULT_ROPE_OVERRIDE=""
fi
export ROPE_OVERRIDE="${ROPE_OVERRIDE-${DEFAULT_ROPE_OVERRIDE}}"

echo "====================================="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "WORKDIR=${PWD}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "ALL_NODES=${ALL_NODES}"
echo "IMAGE=${DYNAMO_IMAGE}"
echo "MODEL=${MODEL_NAME}"
echo "TP=${TP}  CONTEXT_LEN=${CONTEXT_LEN}  KV_DTYPE=${KV_DTYPE}"
echo "MAX_RUNNING=${MAX_RUNNING}  MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS}  CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}"
echo "MEM_FRAC=${MEM_FRAC}  ATTN_BACKEND=${ATTN_BACKEND}"
echo "SPEC_ALGO=${SPEC_ALGO:-<off>}  SPEC_STEPS=${SPEC_STEPS}  SPEC_DRAFT=${SPEC_DRAFT:-<n/a>}"
echo "ROPE_OVERRIDE=${ROPE_OVERRIDE:-<none, native ctx>}"
echo "HF_TOKEN_SET=$( [ -n "${HF_TOKEN}" ] && echo yes || echo no )"
echo "====================================="

check_health() {
    curl -sf "http://${HEAD_NODE}:${LLM_PORT}/health"
}

worker_ready() {
    check_health | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('instances') else 1)"
}

wait_for() {
    local check_cmd=$1
    local attempts=$2
    local delay=$3
    for i in $(seq 1 "$attempts"); do
        if "$check_cmd" >/dev/null; then
            return 0
        fi
        sleep "$delay"
    done
    "$check_cmd" >/dev/null
}

WORKER_PIDS=()
cleanup() {
    exit_code=$?
    log "Stopping Dynamo job"
    if [ ${#WORKER_PIDS[@]} -gt 0 ]; then
        kill "${WORKER_PIDS[@]}" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    rm -f "${READY_MARKER}"
    if [ "$exit_code" -ne 0 ]; then
        log "Job failed."
    fi
    exit "$exit_code"
}
trap cleanup EXIT

log "Starting Dynamo frontend on ${HEAD_NODE}"
podman_dynamo python3 -m dynamo.frontend \
    --discovery-backend file > "${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!

log "Waiting for frontend"
wait_for check_health 30 5

build_worker_args() {
    local rank=$1
    local args=(
        --model-path "${MODEL_NAME}"
        --discovery-backend file
        --tensor-parallel-size "${TP}"
        --context-length "${CONTEXT_LEN}"
        --max-running-requests "${MAX_RUNNING}"
        --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
        --mem-fraction-static "${MEM_FRAC}"
        --attention-backend "${ATTN_BACKEND}"
        --trust-remote-code
    )
    if [ "${KV_DTYPE}" != "auto" ]; then
        args+=(--kv-cache-dtype "${KV_DTYPE}")
    fi
    if [ -n "${ROPE_OVERRIDE}" ]; then
        args+=(--json-model-override-args "${ROPE_OVERRIDE}")
    fi
    if [ -n "${SPEC_ALGO}" ]; then
        args+=(--speculative-algorithm "${SPEC_ALGO}"
               --speculative-num-steps "${SPEC_STEPS}")
        if [ -n "${SPEC_DRAFT}" ]; then
            args+=(--speculative-draft-model-path "${SPEC_DRAFT}")
        fi
    fi
    if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
        args+=(--dist-init-addr "${HEAD_NODE}:50000"
               --nnodes "${SLURM_JOB_NUM_NODES}"
               --node-rank "${rank}")
    fi
    printf '%q ' "${args[@]}"
}

start_worker() {
    local rank=$1
    local node=$2
    local worker_args
    worker_args=$(build_worker_args "$rank")
    log "Starting Dynamo SGLang worker ${rank} -> ${node}"
    local worker_log="${WORKER_LOG_PREFIX}-${rank}-${node}.log"
    srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=4 -w "$node" \
        bash -lc "podman_dynamo python3 -m dynamo.sglang ${worker_args}" \
        > "${worker_log}" 2>&1 &
    WORKER_PIDS+=("$!")
}

for rank in "${!NODES[@]}"; do
    start_worker "$rank" "${NODES[$rank]}"
    sleep 5
done

log "Waiting for Dynamo worker registration (up to ~30 min for 1M ctx warmup)"
wait_for worker_ready 180 10

cat > "${READY_MARKER}" <<EOF
HEAD_NODE=${HEAD_NODE}
LLM_PORT=${LLM_PORT}
BASE_URL=http://${HEAD_NODE}:${LLM_PORT}
MODEL_NAME=${MODEL_NAME}
CONTEXT_LEN=${CONTEXT_LEN}
TP=${TP}
KV_DTYPE=${KV_DTYPE}
SLURM_JOB_ID=${SLURM_JOB_ID}
EOF

log "Dynamo SGLang ready. Endpoint: http://${HEAD_NODE}:${LLM_PORT}/v1"
log "Ready marker: ${READY_MARKER}"
log "Run benchmark:  HEAD_NODE=${HEAD_NODE} MODEL_NAME=${MODEL_NAME} ./bench_sglang_qwen3.6_1m.sh"
log "Server will stay up until walltime expires or job is cancelled."

# Stay alive so the OpenAI endpoint remains reachable for the bench driver.
wait
