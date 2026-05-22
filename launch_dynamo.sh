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

set -eE

export HF_HOME=$SCRATCH/cache
export TVM_FFI_CACHE_DIR=$SCRATCH/.cache

# HF_TOKEN is not needed for the default public model:
#   Qwen/Qwen3-0.6B
#
# HF_TOKEN is needed for gated/private/license models, for example:
#   meta-llama/Llama-3.1-8B-Instruct
#
# Recommended:
#  export HF_TOKEN=hf_your_token_here
# or:
#   echo "hf_your_token_here" > ~/.hf_token
#   chmod 600 ~/.hf_token
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null || true)}"
export DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2"
export DYNAMO_STORE="$SCRATCH/dynamo_store_kv_${SLURM_JOB_ID}"
FRONTEND_LOG="logs/dynamo-frontend-${SLURM_JOB_ID}.log"
WORKER_LOG_PREFIX="logs/dynamo-worker-${SLURM_JOB_ID}"

mkdir -p logs "$SCRATCH"/root "$HF_HOME" "$TVM_FFI_CACHE_DIR" "$DYNAMO_STORE"
rm -f "logs/dynamo-response-${SLURM_JOB_ID}.json" "logs/dynamo-response-${SLURM_JOB_ID}.json.tmp"
rm -f "${FRONTEND_LOG}" "${WORKER_LOG_PREFIX}"*.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

podman_dynamo() {
    # NCCL_DEBUG is intentionally unset by default.
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
    -e "TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR}" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    -e "HF_HOME=${HF_HOME}" \
    -w /workdir \
    ${DYNAMO_IMAGE} \
    "$@"
}
export -f podman_dynamo

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALL_NODES="${NODES[*]}"
HEAD_NODE="${NODES[0]}"

export LLM_HOST=0.0.0.0
export LLM_PORT=8000

# Public smoke-test model.
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
# export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
# export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-Next}"
# export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-Next-FP8}"
# export MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

# Larger examples for 1 node / 2 nodes / 3+ nodes
# export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
# export MODEL_NAME="openai/gpt-oss-120b"
# export MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

# Gated Hugging Face example: needs HF_TOKEN and accepted license.
# export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# Or submit without editing:
# MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct sbatch launch_dynamo.sh

# Sets tensor parallel to number of GPUs in the job: 4 x node count.
export TP=$(( 4 * SLURM_JOB_NUM_NODES ))

echo "====================================="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "WORKDIR=${PWD}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"
echo "IMAGE=${DYNAMO_IMAGE}"
echo "MODEL=${MODEL_NAME}"
echo "TENSOR_PARALLEL_SIZE=${TP}"
if [ -n "${HF_TOKEN}" ]; then
    echo "HF_TOKEN_SET=yes"
else
    echo "HF_TOKEN_SET=no"
fi
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
    if [ "$exit_code" -ne 0 ]; then
        log "Job failed."
    fi
    exit "$exit_code"
}
trap cleanup EXIT

log "All nodes: ${ALL_NODES}"
log "Head node: ${HEAD_NODE}"
log "Model: ${MODEL_NAME}"
log "Tensor parallel size: ${TP}"

log "Starting Dynamo frontend on ${HEAD_NODE}"
podman_dynamo python3 -m dynamo.frontend \
    --discovery-backend file > "${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!

log "Waiting for frontend"
wait_for check_health 30 5

start_worker() {
    local rank=$1
    local node=$2
    local dist_args=""

    if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
        dist_args="--dist-init-addr ${HEAD_NODE}:50000 --nnodes ${SLURM_JOB_NUM_NODES} --node-rank ${rank}"
    fi

    log "Starting Dynamo SGLang worker ${rank} -> ${node}"
    local worker_log="${WORKER_LOG_PREFIX}-${rank}-${node}.log"
    srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=4 -w "$node" \
        bash -lc "
            podman_dynamo python3 -m dynamo.sglang \
            --model-path ${MODEL_NAME} \
            --discovery-backend file \
            --tensor-parallel-size ${TP} \
            ${dist_args}
    " > "${worker_log}" 2>&1 &
    WORKER_PIDS+=("$!")
}

for rank in "${!NODES[@]}"; do
    start_worker "$rank" "${NODES[$rank]}"
    sleep 5
done

log "Waiting for Dynamo worker registration"
wait_for worker_ready 90 10

log "Running test prompt"
RESPONSE_FILE="logs/dynamo-response-${SLURM_JOB_ID}.json"
RESPONSE_TMP="${RESPONSE_FILE}.tmp"

for i in {1..30}; do
    HTTP_CODE=$(curl -sS -o "${RESPONSE_TMP}" -w "%{http_code}" \
        "http://${HEAD_NODE}:${LLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\":\"${MODEL_NAME}\",
            \"messages\":[{\"role\":\"user\",\"content\":\"Explain what NVIDIA Dynamo does on top of SGLang.\"}],
            \"max_tokens\":1000
        }")

    if [ "${HTTP_CODE}" = "200" ]; then
        python3 -m json.tool "${RESPONSE_TMP}" > "${RESPONSE_FILE}"
        rm -f "${RESPONSE_TMP}"
        break
    fi

    log "Chat endpoint not ready yet, HTTP ${HTTP_CODE}. Retrying..."
    sleep 10
done

if [ ! -s "${RESPONSE_FILE}" ]; then
    echo "Chat completion failed. Last response:"
    cat "${RESPONSE_TMP}" 2>/dev/null || true
    exit 1
fi

log "Response saved to ${RESPONSE_FILE}"
