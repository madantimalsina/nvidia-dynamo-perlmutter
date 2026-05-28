#!/usr/bin/env bash
# Drives prefill / single-request decode / batched decode peak measurements
# against a running Dynamo+SGLang OpenAI-compatible endpoint, and prints a
# summary in the same shape as the vLLM baseline:
#
#   Prefill peak              : <tok/s>
#   Decode single-request     : <tok/s>
#   Decode batched peak       : <tok/s>  (at concurrency C=<N>)
#
# Usage:
#   # 1) After the launcher job is healthy, find HEAD_NODE either from
#   #    logs/dynamo-ready-<jobid>.env or from squeue.
#   READY=logs/dynamo-ready-<jobid>.env
#   source "$READY"
#   ./bench_qwen3.6_1m.sh
#
#   # Or pass it explicitly:
#   HEAD_NODE=nid001234 MODEL_NAME=Qwen/Qwen3.6-27B ./bench_qwen3.6_1m.sh
#
# Override which sweeps run by setting any of:
#   PREFILL_INPUT_LENS, DECODE_CONCURRENCY, DECODE_OUTPUT_LEN, etc.

set -euo pipefail

# ── Endpoint ──────────────────────────────────────────────────────────────────
if [ -z "${HEAD_NODE:-}" ] && [ -z "${BASE_URL:-}" ]; then
    latest_ready=$(ls -t logs/dynamo-ready-*.env 2>/dev/null | head -n 1 || true)
    if [ -n "${latest_ready}" ]; then
        echo "[bench] sourcing ${latest_ready}"
        # shellcheck disable=SC1090
        source "${latest_ready}"
    fi
fi

: "${LLM_PORT:=8000}"
: "${BASE_URL:=http://${HEAD_NODE}:${LLM_PORT}}"
: "${MODEL_NAME:=Qwen/Qwen3.6-27B}"

# ── Sweep configuration ───────────────────────────────────────────────────────
# Prefill peak: long input, 1 output token, low concurrency.
# Adjust DOWNWARD if 1M-ctx prefill OOMs at the largest size.
: "${PREFILL_INPUT_LENS:=32768 131072 262144 524288}"
: "${PREFILL_OUTPUT_LEN:=1}"
: "${PREFILL_NUM_PROMPTS:=8}"
: "${PREFILL_CONCURRENCY:=2}"

# Decode single request: small prompt, generate, concurrency = 1.
: "${DECODE_SINGLE_INPUT_LEN:=128}"
: "${DECODE_SINGLE_OUTPUT_LEN:=1024}"
: "${DECODE_SINGLE_NUM_PROMPTS:=8}"

# Decode batched: small prompt, short output, sweep concurrency.
: "${DECODE_BATCH_INPUT_LEN:=128}"
: "${DECODE_BATCH_OUTPUT_LEN:=256}"
: "${DECODE_CONCURRENCY:=4 8 16 32 64 128}"

# ── Output layout ─────────────────────────────────────────────────────────────
STAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="logs/bench-${STAMP}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.txt"

echo "[bench] BASE_URL=${BASE_URL}"
echo "[bench] MODEL=${MODEL_NAME}"
echo "[bench] OUT_DIR=${OUT_DIR}"

# Sanity check the endpoint is reachable.
if ! curl -sf "${BASE_URL}/health" >/dev/null; then
    echo "[bench] ERROR: ${BASE_URL}/health is not reachable. Is the launcher job running?" >&2
    exit 1
fi

# ── Pick a benchmark client ───────────────────────────────────────────────────
# Prefer sglang.bench_serving; fall back to `vllm bench serve`.
BENCH_KIND=""
if python3 -c "import sglang.bench_serving" 2>/dev/null; then
    BENCH_KIND="sglang"
elif command -v vllm >/dev/null 2>&1; then
    BENCH_KIND="vllm"
else
    cat >&2 <<'EOF'
[bench] ERROR: need either sglang.bench_serving or the vllm CLI installed.
        On the head node, run this from inside the Dynamo container, e.g.:
          podman-hpc exec <dynamo-container> python3 -m sglang.bench_serving ...
        or:  pip install --user sglang && rerun.
EOF
    exit 1
fi
echo "[bench] using ${BENCH_KIND} client"

# Run one benchmark, write the raw log, echo the path.
run_bench() {
    local tag=$1 input_len=$2 output_len=$3 num_prompts=$4 concurrency=$5
    local log="${OUT_DIR}/${tag}.log"
    if [ "${BENCH_KIND}" = "sglang" ]; then
        python3 -m sglang.bench_serving \
            --backend sglang-oai \
            --base-url "${BASE_URL}" \
            --model "${MODEL_NAME}" \
            --dataset-name random \
            --random-input-len "${input_len}" \
            --random-output-len "${output_len}" \
            --random-range-ratio 1.0 \
            --num-prompts "${num_prompts}" \
            --max-concurrency "${concurrency}" \
            > "${log}" 2>&1 || true
    else
        vllm bench serve \
            --backend openai-chat \
            --base-url "${BASE_URL}" \
            --model "${MODEL_NAME}" \
            --dataset-name random \
            --random-input-len "${input_len}" \
            --random-output-len "${output_len}" \
            --num-prompts "${num_prompts}" \
            --max-concurrency "${concurrency}" \
            > "${log}" 2>&1 || true
    fi
    echo "${log}"
}

# Extract "Input token throughput" tok/s from the bench output (both clients
# print this label).
extract_input_tput() {
    grep -Ei "Input token throughput" "$1" | tail -n 1 \
        | grep -oE '[0-9]+\.[0-9]+' | head -n 1
}
extract_output_tput() {
    grep -Ei "Output token throughput" "$1" | tail -n 1 \
        | grep -oE '[0-9]+\.[0-9]+' | head -n 1
}

# ── A) Prefill peak ───────────────────────────────────────────────────────────
echo "[bench] === A) prefill peak sweep ==="
best_prefill=0
best_prefill_len=0
for inlen in ${PREFILL_INPUT_LENS}; do
    echo "[bench]  prefill input_len=${inlen} concurrency=${PREFILL_CONCURRENCY}"
    log=$(run_bench "prefill-in${inlen}" "${inlen}" "${PREFILL_OUTPUT_LEN}" \
                    "${PREFILL_NUM_PROMPTS}" "${PREFILL_CONCURRENCY}")
    tput=$(extract_input_tput "${log}" || true)
    echo "[bench]    -> input_throughput=${tput:-NA} tok/s  (log=${log})"
    if [ -n "${tput:-}" ]; then
        better=$(python3 -c "print(1 if float('${tput}') > float('${best_prefill}') else 0)")
        if [ "${better}" = "1" ]; then
            best_prefill="${tput}"
            best_prefill_len="${inlen}"
        fi
    fi
done

# ── B) Decode, single request ─────────────────────────────────────────────────
echo "[bench] === B) decode single-request ==="
log=$(run_bench "decode-single" "${DECODE_SINGLE_INPUT_LEN}" \
                "${DECODE_SINGLE_OUTPUT_LEN}" "${DECODE_SINGLE_NUM_PROMPTS}" 1)
decode_single=$(extract_output_tput "${log}" || true)
echo "[bench]    -> output_throughput=${decode_single:-NA} tok/s  (log=${log})"

# ── C) Decode, batched peak ───────────────────────────────────────────────────
echo "[bench] === C) decode batched peak sweep ==="
best_decode=0
best_decode_c=0
for c in ${DECODE_CONCURRENCY}; do
    np=$(( c * 4 ))
    echo "[bench]  decode concurrency=${c} num_prompts=${np}"
    log=$(run_bench "decode-c${c}" "${DECODE_BATCH_INPUT_LEN}" \
                    "${DECODE_BATCH_OUTPUT_LEN}" "${np}" "${c}")
    tput=$(extract_output_tput "${log}" || true)
    echo "[bench]    -> output_throughput=${tput:-NA} tok/s  (log=${log})"
    if [ -n "${tput:-}" ]; then
        better=$(python3 -c "print(1 if float('${tput}') > float('${best_decode}') else 0)")
        if [ "${better}" = "1" ]; then
            best_decode="${tput}"
            best_decode_c="${c}"
        fi
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
{
    echo "Dynamo + SGLang benchmark  ${STAMP}"
    echo "BASE_URL=${BASE_URL}"
    echo "MODEL=${MODEL_NAME}"
    echo "CONTEXT_LEN=${CONTEXT_LEN:-unknown}  TP=${TP:-unknown}  KV_DTYPE=${KV_DTYPE:-unknown}"
    echo
    printf "Prefill peak              : %s tok/s  (at input_len=%s)\n" \
           "${best_prefill}" "${best_prefill_len}"
    printf "Decode single-request     : %s tok/s\n" "${decode_single:-NA}"
    printf "Decode batched peak       : %s tok/s  (at concurrency C=%s)\n" \
           "${best_decode}" "${best_decode_c}"
} | tee "${SUMMARY}"

echo
echo "[bench] summary -> ${SUMMARY}"
echo "[bench] raw logs -> ${OUT_DIR}/"
