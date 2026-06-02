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

# Unified Dynamo launcher. Dispatches on $BACKEND (sglang|vllm).
# MAX_MODEL_LEN is applied to both backends (sglang --context-length /
# vllm --max-model-len); required for models whose native ctx exceeds GPU memory.
#
#   sbatch launch_dynamo.sh                                                       # sglang, 1 node, Qwen3-0.6B
#   BACKEND=vllm sbatch launch_dynamo.sh                                          # vllm,   1 node
#   MODEL_NAME=Qwen/Qwen3.6-27B MAX_MODEL_LEN=8192 sbatch launch_dynamo.sh        # 27B, sglang, ctx=8192
#   MODEL_NAME=Qwen/Qwen3.6-27B MAX_MODEL_LEN=8192 BACKEND=vllm sbatch launch_dynamo.sh
#   MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct BACKEND=vllm \
#       sbatch --nodes=2 launch_dynamo.sh                                         # vllm, 2 nodes, gated model

set -eE

# ---- Backend dispatch ----
export BACKEND="${BACKEND:-sglang}"
case "$BACKEND" in
    sglang) DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1" ;;
    vllm)   DYNAMO_IMAGE="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1"   ;;
    *) echo "ERROR: BACKEND must be 'sglang' or 'vllm', got '$BACKEND'"; exit 2 ;;
esac
export DYNAMO_IMAGE

export HF_HOME=$SCRATCH/cache
export TVM_FFI_CACHE_DIR=$SCRATCH/.cache
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null || true)}"
export DYNAMO_STORE="$SCRATCH/dynamo_store_kv_${SLURM_JOB_ID}"

# Backend-specific log dir: log_sglang/ or log_vllm/. The SBATCH -o/-e files
# above still land in logs/ (Slurm directives are static); everything the
# launcher writes (frontend, worker, response) goes under $LOGDIR.
LOGDIR="log_${BACKEND}"
FRONTEND_LOG="${LOGDIR}/dynamo-frontend-${SLURM_JOB_ID}.log"
WORKER_LOG_PREFIX="${LOGDIR}/dynamo-worker-${SLURM_JOB_ID}"

mkdir -p logs "$LOGDIR" "$SCRATCH"/root "$HF_HOME" "$TVM_FFI_CACHE_DIR" "$DYNAMO_STORE"
rm -f "${LOGDIR}/dynamo-response-${SLURM_JOB_ID}.json" "${LOGDIR}/dynamo-response-${SLURM_JOB_ID}.json.tmp"
rm -f "${FRONTEND_LOG}" "${WORKER_LOG_PREFIX}"*.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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
    -e "TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR}" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    -e "HF_HOME=${HF_HOME}" \
    -e "DYN_DISCOVERY_BACKEND=file" \
    -e "DYN_FILE_KV=/tmp/dynamo_store_kv" \
    -e "DYN_REQUEST_PLANE=tcp" \
    -e "DYN_EVENT_PLANE=zmq" \
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
export MASTER_PORT="${MASTER_PORT:-29500}"            # vllm torch.distributed bootstrap
export SGLANG_DIST_PORT="${SGLANG_DIST_PORT:-50000}"  # sglang --dist-init-addr port

# Public smoke-test model — common to both backends.
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"

# Context length cap. Applied to BOTH backends:
#   sglang -> --context-length
#   vllm   -> --max-model-len
# Required for models whose native max ctx exceeds GPU memory (e.g. Qwen3.6-27B native=256k).
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# TP = total GPUs across all allocated nodes (4 GPUs/node).
export TP=$(( 4 * SLURM_JOB_NUM_NODES ))

echo "====================================="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "WORKDIR=${PWD}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"
echo "BACKEND=${BACKEND}"
echo "LOGDIR=${LOGDIR}"
echo "IMAGE=${DYNAMO_IMAGE}"
echo "MODEL=${MODEL_NAME}"
echo "TENSOR_PARALLEL_SIZE=${TP}"
echo "NUM_NODES=${SLURM_JOB_NUM_NODES}"
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
    local check_cmd=$1 attempts=$2 delay=$3
    for i in $(seq 1 "$attempts"); do
        if "$check_cmd" >/dev/null; then return 0; fi
        sleep "$delay"
    done
    "$check_cmd" >/dev/null
}

WORKER_PIDS=()
cleanup() {
    exit_code=$?
    log "Stopping Dynamo job"
    [ ${#WORKER_PIDS[@]} -gt 0 ] && kill "${WORKER_PIDS[@]}" 2>/dev/null || true
    [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    [ "$exit_code" -ne 0 ] && log "Job failed."
    exit "$exit_code"
}
trap cleanup EXIT

log "All nodes: ${ALL_NODES}"
log "Head node: ${HEAD_NODE}"
log "Backend:   ${BACKEND}"
log "Model:     ${MODEL_NAME}"
log "TP:        ${TP}"

# ---- Frontend (identical for both backends) ----
log "Starting Dynamo frontend on ${HEAD_NODE}"
podman_dynamo python3 -m dynamo.frontend \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq > "${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!

log "Waiting for frontend"
wait_for check_health 30 5

# ---- Worker (backend-specific argv) ----
build_worker_cmd() {
    local rank=$1

    if [ "$BACKEND" = "sglang" ]; then
        local dist_args=""
        if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
            dist_args="--dist-init-addr ${HEAD_NODE}:${SGLANG_DIST_PORT} --nnodes ${SLURM_JOB_NUM_NODES} --node-rank ${rank}"
        fi
        echo "python3 -m dynamo.sglang \
            --model-path ${MODEL_NAME} \
            --discovery-backend file \
            --request-plane tcp \
            --event-plane zmq \
            --tensor-parallel-size ${TP} \
            --context-length ${MAX_MODEL_LEN} \
            ${dist_args}"
    else
        # vLLM uses torch.distributed for multi-node; --headless on rank > 0.
        local extra=""
        if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
            extra="--nnodes ${SLURM_JOB_NUM_NODES} --node-rank ${rank} --master-addr ${HEAD_NODE} --master-port ${MASTER_PORT}"
            [ "$rank" -ne 0 ] && extra="${extra} --headless"
        fi
        echo "python3 -m dynamo.vllm \
            --model ${MODEL_NAME} \
            --discovery-backend file \
            --request-plane tcp \
            --event-plane zmq \
            --tensor-parallel-size ${TP} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-num-seqs ${MAX_NUM_SEQS} \
            --enforce-eager \
            ${extra}"
    fi
}

start_worker() {
    local rank=$1 node=$2
    local worker_log="${WORKER_LOG_PREFIX}-${rank}-${node}.log"
    local cmd
    cmd=$(build_worker_cmd "$rank")
    log "Starting Dynamo ${BACKEND} worker ${rank} -> ${node}"
    srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=4 -w "$node" \
        bash -lc "podman_dynamo ${cmd}" > "${worker_log}" 2>&1 &
    WORKER_PIDS+=("$!")
}

for rank in "${!NODES[@]}"; do
    start_worker "$rank" "${NODES[$rank]}"
    sleep 5
done

log "Waiting for Dynamo worker registration"
wait_for worker_ready 180 10   # 30 min — gives headroom for 27B-class model load + CUDA graph capture

log "Running test prompt"
RESPONSE_FILE="${LOGDIR}/dynamo-response-${SLURM_JOB_ID}.json"
RESPONSE_TMP="${RESPONSE_FILE}.tmp"

PROMPT="Explain what NVIDIA Dynamo does on top of ${BACKEND}."

for i in {1..30}; do
    HTTP_CODE=$(curl -sS -o "${RESPONSE_TMP}" -w "%{http_code}" \
        "http://${HEAD_NODE}:${LLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\":\"${MODEL_NAME}\",
            \"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],
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
