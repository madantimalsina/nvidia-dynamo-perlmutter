#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH -A nstaff
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
HF_TOKEN="$(cat ~/.hf_token 2>/dev/null || true)"
export HF_TOKEN
# export SGLANG_IMAGE="lmsysorg/sglang:latest-runtime"
export SGLANG_IMAGE="lmsysorg/sglang:latest-cu130-runtime"

# Logging function
log() {
    GR='\033[0;32m'
    BL='\033[0;34m'
    NC='\033[0m'
    echo -e "${GR}[$(date '+%Y-%m-%d %H:%M:%S')] ${BL} $* ${NC}"
}

podman_llm() {
    podman-hpc run \
    --rm \
    -u 0 \
    --gpu \
    --nccl-cu13 \
    --net host \
    --ipc=host \
    -v "$SCRATCH"/root:/root \
    -v "$SCRATCH":"$SCRATCH" \
    -v "${PWD}:/workdir" \
    -e "NCCL_DEBUG=INFO" \
    -e "TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR}" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    -e "HF_HOME=${HF_HOME}" \
    -w /workdir \
    ${SGLANG_IMAGE} \
    "$@"
}
export -f podman_llm

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2)


export LLM_HOST=0.0.0.0
export LLM_PORT=8000

# 2 nodes - 8 TP
# export MODEL_NAME="Qwen/Qwen3.5-27B"
# export NICE_MODEL_NAME="qwen3.5"

# 2 nodes - 8 TP
# export MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
# export NICE_MODEL_NAME="Nemotron3"

# 1 node - 4 TP
#export MODEL_NAME="openai/gpt-oss-120b"
#export NICE_MODEL_NAME="gpt-oss-120b"

# 2 nodes - 8 TP
# export MODEL_NAME="Qwen/Qwen3-Coder-Next"
# export NICE_MODEL_NAME="qwen3-coder"

# 1 node - 4TP
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export NICE_MODEL_NAME="Mistral-7B"

# Sets tensor paralell to number of GPUs in the job 4x#Nodes
export TP=$(( 4 * SLURM_JOB_NUM_NODES ))

echo "$WORKER_NODES"
if [ -n "$WORKER_NODES" ]; then
    count=0
    log "Starting head node ${count} -> ${HEAD_NODE}"
    for worker in $ALL_NODES; do
        log "Starting worker ${count} -> ${worker}"
        srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=4 -w "$worker" \
            bash -lc "
                podman_llm python3 -m sglang.launch_server \
                --dist-init-addr ${HEAD_NODE}:50000 \
                --model-path ${MODEL_NAME} --served-model-name ${NICE_MODEL_NAME} \
                --host ${LLM_HOST} --port ${LLM_PORT} \
                --tp ${TP} \
                --trust-remote-code \
                --nnodes ${SLURM_JOB_NUM_NODES} --node-rank ${count}
        " &
        sleep 5
        count=$((count+1))
    done;
else
    log "No worker nodes allocated (single-node cluster) ${HEAD_NODE}"
    srun --nodes=1 --ntasks=1 --cpus-per-task=128 --gpus-per-task=4 \
            bash -lc "
                podman_llm python3 -m sglang.launch_server \
                --model-path ${MODEL_NAME} --host ${LLM_HOST} --port ${LLM_PORT} \
                --tp ${TP}
        " &
fi


wait
