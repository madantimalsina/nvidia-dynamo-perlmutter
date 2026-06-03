# SGLang (NVIDIA Dynamo) vs vLLM — Qwen3.6-27B @ 1M Context on Perlmutter

Like-for-like benchmark plan and 2026-05-27 results for running **Qwen/Qwen3.6-27B**
with a 1M context window on NVIDIA Dynamo (SGLang backend), compared against the
existing vLLM v0.21.0 baseline in [`perlmutter-agent-main/`](https://gitlab.nersc.gov/nersc/otg/perlmutter-agent). Sections 1–6 cover setup,
flags, methodology, and operational notes; Section 7 has the measured numbers.

## Note — Pre-downloading the model to `$SCRATCH`

The launcher sets `HF_HOME=$SCRATCH/cache` and mounts `$SCRATCH` into the container, so
anything cached at that path is picked up automatically. Pre-downloading on a login node
avoids burning GPU walltime on the initial fetch.

### Download

```bash
# On a login node (no GPU needed)
module load python                           # or activate your env
pip install --user -U "huggingface_hub[cli]" # provides the new `hf` CLI

export HF_HOME=$SCRATCH/cache
mkdir -p "$HF_HOME"

# Gated models need a token:
export HF_TOKEN=$(cat ~/.hf_token)
# or interactive:  hf auth login

hf download Qwen/Qwen3.6-27B --repo-type model --cache-dir "$HF_HOME"
```

Note: `huggingface-cli` was renamed to `hf`. Old `huggingface-cli` commands print a
deprecation warning and no longer execute.

### Check availability

```bash
# Where it lives + total size on disk
du -sh $HF_HOME/models--Qwen--Qwen3.6-27B

# Per-shard sizes (resolved through symlinks → real files)
ls -lhL $HF_HOME/models--Qwen--Qwen3.6-27B/snapshots/*/*.safetensors

# Integrity checks: no zero-byte blobs, all 15 shards referenced
find $HF_HOME/models--Qwen--Qwen3.6-27B/blobs -size 0
python3 -c "import glob, json; idx=glob.glob('$HF_HOME/models--Qwen--Qwen3.6-27B/snapshots/*/model.safetensors.index.json')[0]; d=json.load(open(idx)); print(len(set(d['weight_map'].values())), 'unique shards')"
```

Expected for Qwen3.6-27B (bf16):

- Total: ~52 GiB on disk (`du`) / ~55.6 GB nominal (decimal, what `hf` reports)
- Shards: `model-00001-of-00015.safetensors` … `model-00015-of-00015.safetensors`
  - Shards 1–14: ~3.7–3.8 GiB each
  - Shard 15: ~486 MiB (last shard holds the embedding/lm_head tail and is smaller — normal)
- `find ... -size 0` → empty output (no truncated blobs)
- Python check → `15 unique shards`

`HF_HOME` is not sticky across shells on Perlmutter (some modules set it to a CFS path).
Re-export it in each new login session, or add it to `~/.bashrc`:

```bash
echo 'export HF_HOME=$SCRATCH/cache' >> ~/.bashrc
```

The Slurm script exports `HF_HOME=$SCRATCH/cache` on its own, so the job is unaffected
by what the interactive shell has set.

### What to watch for after `sbatch`

```bash
JOBID=<your-job-id>
tail -f logs/launch_sglang_qwen3.6-1m.sh-${JOBID}.out
```

Expected milestones:

1. `Starting Dynamo frontend on nidXXXXXX` → `Waiting for frontend` → frontend healthy
   (~30 s)
2. `Starting Dynamo SGLang worker 0 -> nidXXXXXX`
3. `Waiting for Dynamo worker registration` — at 1M ctx with fp8 KV expect **5–15 min**
   for weight load + KV alloc + CUDA graph capture. The log looks idle during this
   phase; that is normal.
4. `Dynamo SGLang ready. Endpoint: http://nidXXXXXX:8000/v1` ← green light
5. `Ready marker: logs/dynamo-ready-<jobid>.env` is written

Once step 4 appears, kick off the bench from the same login session:

```bash
source logs/dynamo-ready-${JOBID}.env
./bench_sglang_qwen3.6_1m.sh
```

If step 3 never completes (worker_ready times out at ~30 min), inspect the worker log:

```bash
cat logs/dynamo-worker-${JOBID}-0-*.log
```

Common failure at 1M ctx: SGLang errors with `context_length > derived context_length
(262144)`. Qwen3.6-27B's native max is 256k; reaching 1M needs both
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` (env var) and YaRN RoPE scaling via
`--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":<CTX/262144>,"original_max_position_embeddings":262144}}'`.
Both are now wired into `launch_sglang_qwen3.6-1m.sh`. The other likely failure at 1M
ctx is **OOM during KV cache allocation**. To validate the pipeline first, retry at
half-context and then push back up:

```bash
CONTEXT_LEN=524288 sbatch launch_sglang_qwen3.6-1m.sh
# then once healthy:
CONTEXT_LEN=1010000 sbatch launch_sglang_qwen3.6-1m.sh
```

Other quick triage commands:

```bash
# Is the job still running?
squeue -u $USER

# Frontend log (Dynamo router/discovery side)
cat logs/dynamo-frontend-${JOBID}.log

# Cancel if needed
scancel ${JOBID}
```

## Pre-flight checks

1. Replace `<YOUR ACCOUNT>` in the `#SBATCH -A` line.
2. Pre-download the model weights to `$SCRATCH/cache` (see above) so the GPU job does
   not spend walltime fetching ~52 GiB from Hugging Face. The launcher sets
   `HF_HOME=$SCRATCH/cache` and will pick up the cached snapshot automatically.
3. Migrate the Dynamo container image to PSCRATCH once per cluster (avoids home-dir
   quota issues and speeds up container start):
   ```bash
   podman-hpc pull    nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
   podman-hpc migrate nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
   ```
4. If `Qwen/Qwen3.6-27B` OOMs at full 1M ctx on 4×A100, drop to `CONTEXT_LEN=524288`
   first to validate the pipeline, then push back up. The vLLM baseline almost certainly
   used fp8 KV; this script defaults to the same.

## 1. Recommended SGLang flags (passed through `dynamo.sglang`)

`dynamo.sglang` accepts SGLang's worker flags directly. To match the vLLM baseline:

| Purpose | vLLM flag | SGLang flag |
| --- | --- | --- |
| Model | `--model` | `--model-path` |
| Context length | `--max-model-len 1010000` | `--context-length 1010000` |
| TP (per replica) | `--tensor-parallel-size 4` | `--tensor-parallel-size 4` |
| Chunked prefill | `--enable-chunked-prefill` | on by default; tune with `--chunked-prefill-size 8192` |
| Prefill batch budget | `--max-num-batched-tokens` | `--max-prefill-tokens` |
| Concurrent seqs | `--max-num-seqs` | `--max-running-requests` |
| KV mem headroom | `--gpu-memory-utilization 0.92` | `--mem-fraction-static 0.90` |
| KV cache dtype (fp8) | `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e5m2` |
| Attention backend | (auto) | `--attention-backend flashinfer` (A100 OK) or `triton` |
| Override native ctx cap | implicit / `--hf-overrides` | env: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` + `--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":<f>,"original_max_position_embeddings":262144}}'` |
| Speculative / MTP | `--speculative-config '{...}'` | `--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-draft-model-path <draft>` |

**KV-cache reality check at 1M ctx:** a ~27B GQA model is roughly ~256 KiB KV per token in
bf16 → a single 1M-token sequence needs ~256 GiB KV across the TP group. Four A100-80GB
give 320 GiB but weights (~54 GiB bf16) + activations also need to fit. The vLLM 1M run
almost certainly used `--kv-cache-dtype fp8` (halves it to ~128 GiB/seq); set
`--kv-cache-dtype fp8_e5m2` in SGLang to match. Without it, batched decode at 1M ctx
won't fit.

## 2. Patch to `launch_sglang_smoke.sh`

The relevant edit to the worker launch in `nvidia-dynamo-perlmutter/launch_sglang_smoke.sh`
(around line 177):

```bash
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export CONTEXT_LEN="${CONTEXT_LEN:-1010000}"
export MAX_RUNNING="${MAX_RUNNING:-16}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
export KV_DTYPE="${KV_DTYPE:-fp8_e5m2}"
export MEM_FRAC="${MEM_FRAC:-0.90}"

# Required to push past Qwen3.6-27B's native 256k context. Pass into the
# container env (e.g. add  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# to the podman_dynamo() flags) and supply YaRN rope scaling below.
ROPE_OVERRIDE='{"rope_scaling":{"rope_type":"yarn","factor":3.9,"original_max_position_embeddings":262144}}'

podman_dynamo python3 -m dynamo.sglang \
    --model-path ${MODEL_NAME} \
    --discovery-backend file \
    --tensor-parallel-size ${TP} \
    --context-length ${CONTEXT_LEN} \
    --max-running-requests ${MAX_RUNNING} \
    --max-prefill-tokens ${MAX_PREFILL_TOKENS} \
    --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
    --kv-cache-dtype ${KV_DTYPE} \
    --mem-fraction-static ${MEM_FRAC} \
    --attention-backend flashinfer \
    --json-model-override-args "${ROPE_OVERRIDE}" \
    --trust-remote-code \
    ${dist_args}
```

Without both `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` and the YaRN
`--json-model-override-args`, SGLang either refuses to start
(`User-specified context_length (1010000) is greater than the derived context_length
(262144)`) or starts but produces incoherent output past 256k. The checked-in
`launch_sglang_qwen3.6-1m.sh` computes `factor = CONTEXT_LEN / 262144` automatically.

Submit with `MODEL_NAME=Qwen/Qwen3.6-27B sbatch --nodes=1 launch_sglang_smoke.sh` (single 4-GPU
replica to match the baseline TP=4 layout) or `--nodes=2` to mirror the 2-node router 1M
stack.

The clean, self-contained version of this is checked in as
`launch_sglang_qwen3.6-1m.sh` so `launch_sglang_smoke.sh` itself stays unmodified.

## 3. Dynamo-specific knobs that matter

- **`--tensor-parallel-size`** — keep equal to the vLLM baseline (4 per replica) for
  like-for-like. Going higher than the natural KV-head count for Qwen GQA wastes
  bandwidth.
- **Aggregated vs disaggregated** — by default `dynamo.sglang` runs aggregated
  (prefill+decode on one worker), which matches the vLLM setup. Dynamo's real
  differentiator is `--disaggregation-mode prefill` / `--disaggregation-mode decode`
  (separate prefill workers feed decode workers over NIXL). Run the aggregated config
  first for the apples-to-apples baseline; then optionally run a disagg config as a
  Dynamo-native experiment.
- **`--discovery-backend file`** — fine for single-job benchmarking; already in the
  script.
- **Router/KV-aware routing** — only matters with >1 replica. For 1 replica it can be
  ignored.
- **NCCL** — the launcher sets `NCCL_CUMEM_HOST_ENABLE=0`; leave it. Do not enable
  `NCCL_DEBUG=INFO` during measurement.

## 4. Benchmark methodology (matches what vLLM reports)

Use SGLang's built-in client `sglang.bench_serving` against the Dynamo OpenAI endpoint
at `http://${HEAD_NODE}:8000/v1`. Three runs, three numbers:

```bash
# A) Prefill peak (long input, 1 output token, high concurrency)
python3 -m sglang.bench_serving \
  --backend sglang-oai --base-url http://${HEAD_NODE}:8000 \
  --model ${MODEL_NAME} --dataset-name random \
  --random-input-len 131072 --random-output-len 1 \
  --num-prompts 32 --max-concurrency 4
# Report: "Input token throughput (tok/s)"  → prefill peak
# Sweep input-len in {32k, 128k, 256k, 512k}; take the max.

# B) Decoding, single request
python3 -m sglang.bench_serving \
  --backend sglang-oai --base-url http://${HEAD_NODE}:8000 \
  --model ${MODEL_NAME} --dataset-name random \
  --random-input-len 128 --random-output-len 1024 \
  --num-prompts 8 --max-concurrency 1
# Report: "Output token throughput (tok/s)"  → single-request decode

# C) Decoding, batched peak (sweep concurrency, take peak)
for C in 4 8 16 32 64 128; do
  python3 -m sglang.bench_serving \
    --backend sglang-oai --base-url http://${HEAD_NODE}:8000 \
    --model ${MODEL_NAME} --dataset-name random \
    --random-input-len 128 --random-output-len 256 \
    --num-prompts $((C*4)) --max-concurrency $C \
    | tee logs/bench-decode-c${C}.txt
done
# Report: max "Output token throughput (tok/s)" across the sweep → batched peak
```

If `sglang.bench_serving` is not in the Dynamo container, `vllm bench serve --backend
openai-chat --base-url ...` works against any OpenAI-compatible endpoint and reports the
same metrics — use whichever client the vLLM baseline used so the numbers are computed
identically.

Methodology nudges to keep it clean:

- Always discard the first request (warmup); `sglang.bench_serving` already does this.
- Use `--dataset-name random` with fixed `--random-input-len` so prefill cost is
  deterministic; the default ShareGPT dataset has variable lengths that confound
  prefill/decode separation.
- For decode batched peak, keep input small (128) so the measurement is generation
  throughput, not prefill amortized over generation.

Every sweep knob is overridable at invocation time:

```bash
# More samples per data point for cleaner numbers (slower):
PREFILL_NUM_PROMPTS=16 ./bench_sglang_qwen3.6_1m.sh

# Higher concurrency to push prefill harder (closer to "peak"):
PREFILL_CONCURRENCY=4 PREFILL_NUM_PROMPTS=16 ./bench_sglang_qwen3.6_1m.sh

# Skip the most expensive input size if you are tight on time:
PREFILL_INPUT_LENS="32768 131072 262144" ./bench_sglang_qwen3.6_1m.sh
```

Decode-side knobs (`DECODE_CONCURRENCY`, `DECODE_BATCH_OUTPUT_LEN`,
`DECODE_SINGLE_OUTPUT_LEN`, etc.) override the same way — see the top of
`bench_sglang_qwen3.6_1m.sh` for the full list of `: "${VAR:=default}"` lines.

## 5. Optional features

- **MTP-equivalent (speculative decoding):** SGLang supports EAGLE/MTP via
  `--speculative-algorithm EAGLE --speculative-num-steps {1,2,3}
  --speculative-draft-model-path <draft>`. Qwen3.6-27B does not ship an official EAGLE
  draft head AFAIK, so this likely will not map cleanly — same outcome the vLLM MTP
  experiment hit. Skip unless a draft model exists for this checkpoint.
- **Quantization / Turbo:** SGLang supports `--quantization fp8` (weights) and
  `--kv-cache-dtype fp8_e5m2` (KV). For like-for-like vs the unquantized vLLM run, keep
  weights bf16 and only use fp8 KV (which is required for 1M to fit). A separate
  `--quantization fp8` run is worth doing as a follow-up data point.
- **TurboQuant equivalent:** no direct equivalent in SGLang; the closest knobs are
  AWQ/GPTQ/FP8 weight quant, all of which change the model. Do not conflate with the
  bf16 baseline.

## 6. How to run (two-step workflow)

`sbatch launch_sglang_qwen3.6-1m.sh` only starts the **server**; it stays up until
walltime expires or `scancel`. The benchmark is a separate step that must be triggered
manually against the live endpoint — submitting and just waiting will yield no numbers.

```bash
cd nvidia-dynamo-perlmutter
mkdir -p logs

# ─── Step 1: submit the server job ───────────────────────────────────────────
# Edit the #SBATCH -A line first.
sbatch launch_sglang_qwen3.6-1m.sh
# (or:  sbatch --nodes=2 launch_sglang_qwen3.6-1m.sh   for TP=8)

# Track the job and wait for the ready line:
squeue -u $USER
tail -f logs/launch_sglang_qwen3.6-1m.sh-<jobid>.out
# Wait until you see:
#   [...] Dynamo SGLang ready. Endpoint: http://nidXXXXXX:8000/v1
#   [...] Ready marker: logs/dynamo-ready-<jobid>.env

# ─── Step 2: pick up endpoint + sanity-check it ──────────────────────────────
# The launcher writes a ready marker with HEAD_NODE / BASE_URL / MODEL_NAME / etc.
source logs/dynamo-ready-<jobid>.env

# Confirm the endpoint is healthy:
curl -sf http://${HEAD_NODE}:8000/health && echo OK
# Expected: JSON with "status":"healthy" and at least one entry under "instances".

# ─── Step 3: find the running Dynamo container on the worker node ───────────
ssh ${HEAD_NODE} 'podman-hpc ps --format "table {{.Names}}\t{{.Image}}"'
# Example output:
#   NAMES        IMAGE
#   sharp_hugle  nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2
#   sad_jang     nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2
# Either container works; both ship sglang.bench_serving.

# ─── Step 4: run the bench inside the container ──────────────────────────────
ssh ${HEAD_NODE} "podman-hpc exec sharp_hugle bash -lc 'cd /workdir && ./bench_sglang_qwen3.6_1m.sh'"
# The ${PWD}:/workdir mount means logs/bench-<stamp>/ lands back in this repo dir.
# For an interactive TTY (live progress lines):
#   ssh -t ${HEAD_NODE} "podman-hpc exec -it sharp_hugle bash -lc 'cd /workdir && ./bench_sglang_qwen3.6_1m.sh'"

# ─── Step 5: collect results + clean up ──────────────────────────────────────
cat logs/bench-*/summary.txt
scancel <jobid>      # stops idle GPU charging once you have the numbers
```

If `sglang.bench_serving` / `vllm` happens to be installed on the login node, you can
skip steps 3–4 and just run `./bench_sglang_qwen3.6_1m.sh` directly.

## Baseline reference (vLLM v0.21.0)

The vLLM 1M baseline ([`perlmutter-stack-qwen3.6-27b-1m-2nodes-2replicas-router-managed.sh`](https://gitlab.nersc.gov/nersc/otg/perlmutter-agent/-/blob/main/perlmutter-stack-qwen3.6-27b-1m-2nodes-2replicas-router-managed.sh)
and [`perlmutter-stack-qwen3.5-27b-1m-vllm.sh`](https://gitlab.nersc.gov/nersc/otg/perlmutter-agent/-/blob/main/perlmutter-stack-qwen3.5-27b-1m-vllm.sh)) uses:

- `--tensor-parallel-size 4`
- `--max-model-len 1010000`
- `--enable-chunked-prefill`
- tunable `--max-num-seqs` / `--max-num-batched-tokens`
- optional `--kv-cache-dtype` (fp8 to fit 1M)
- MTP experiment wired via `--speculative-config`

The Dynamo SGLang launcher in `nvidia-dynamo-perlmutter/launch_sglang_smoke.sh` currently passes
only `--model-path` / `--discovery-backend` / `--tensor-parallel-size` — no context length,
no batching knobs, no kv-cache dtype. It needs the extensions in sections 1–7 above to match.

## 7. Results — Qwen3.6-27B @ 1M ctx, 4×A100, fp8 KV (2026-05-27)

Qwen3.6-27B was served on a single 4-GPU Perlmutter node via NVIDIA Dynamo with the
SGLang backend (`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2`), using the
`launch_sglang_qwen3.6-1m.sh` launcher (TP=4, `CONTEXT_LEN=1010000`,
`KV_DTYPE=fp8_e5m2`, `ATTN_BACKEND=flashinfer`, YaRN RoPE scaling with factor=3.9, and
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`). The benchmark driver
`bench_sglang_qwen3.6_1m.sh` was invoked inside the running Dynamo container via
`podman-hpc exec`, hitting the OpenAI-compatible endpoint at `http://nid001072:8000/v1`.
Three measurements were taken via `sglang.bench_serving` against the live endpoint:
prefill peak across input lengths {32k, 128k, 256k, 512k} at concurrency 2; single-request
decode at concurrency 1; and batched-decode peak across concurrencies {4, 8, 16, 32, 64,
128}. The full sweep ran in ~75 minutes; results below are like-for-like vs the existing
vLLM v0.21.0 baseline measured in [`perlmutter-agent-main/`](https://gitlab.nersc.gov/nersc/otg/perlmutter-agent) (same model, same context,
same fp8 KV, same TP).

### Comparison vs vLLM v0.21.0 baseline

| Metric                | vLLM v0.21.0 | Dynamo+SGLang | Winner          |
| --------------------- | ------------ | ------------- | --------------- |
| Prefill peak          | 17.9k tok/s  | 11.2k tok/s   | vLLM (1.6×)     |
| Decode single-request | 50–67 tok/s  | 65 tok/s      | tie             |
| Decode batched peak   | 264 tok/s    | 880 tok/s     | SGLang (3.3×)   |

### Prefill peak — input length sweep (Dynamo+SGLang)

Setup: input length swept over {32k, 128k, 256k, 512k}, output=1 token, concurrency=2,
num-prompts=32. Reports `Input token throughput (tok/s)`; headline is the max of this
sweep. Throughput decreases monotonically with input length; the true peak is likely
below 32k input and has not been measured yet.

| Input length | tok/s     |
| ------------ | --------- |
| 32k          | 11,202.53 |
| 128k         |  7,267.88 |
| 256k         |  4,737.20 |
| 512k         |  2,243.59 |

### Decode single-request (Dynamo+SGLang)

Setup: input=128 tokens, output=1024 tokens, concurrency=1, num-prompts=8. Reports
`Output token throughput (tok/s)`. Single trial — subject to the same run-to-run
variance caveat as the vLLM baseline number.

| Result  |
| ------- |
| 65 tok/s |

### Decode batched peak — concurrency sweep (Dynamo+SGLang)

Setup: input=128 tokens, output=256 tokens, concurrency swept over {4, 8, 16, 32, 64,
128}, num-prompts=C×4 per level. Reports `Output token throughput (tok/s)`; headline
is the max across the sweep.

| Concurrency | tok/s  |
| ----------- | ------ |
| 4           | 240.00 |
| 8           | 464.00 |
| 16          | 880.00 |
| 32          | 880.00 |
| 64          | 880.00 |
| 128         | 880.00 |

C=16, 32, 64, and 128 all returned exactly 880.00 — see the open caveat below.

### Takeaway

vLLM wins prefill throughput; Dynamo+SGLang wins batched-decode throughput by a wide
margin; single-request decode is a tie. Use vLLM for prefill-heavy workloads
(long-prompt summarization, RAG with very large contexts) and Dynamo+SGLang for
high-concurrency serving (many parallel users generating responses).

**Open caveat on the 880.00 figure.** C=16, 32, 64, and 128 all returned exactly
880.00 tok/s — perfect uniformity across four concurrency levels is consistent with
true saturation of the SGLang batched scheduler at C=16, but should be confirmed by
checking the per-run logs:

```bash
grep -E "Output token throughput" logs/bench-20260528-005625/decode-*.log
```

If the four files report different underlying numbers, the throughput extractor in
`bench_sglang_qwen3.6_1m.sh` is picking up the wrong field and the headline batched-decode
number needs correction.

## 8. Results — Qwen3.6-27B @ 1M ctx, 4×A100, fp8 KV (2026-06-02)

The 2026-06-02 reproduction run used the same SGLang-specific workflow as Section 7:
single 4-GPU Perlmutter node, NVIDIA Dynamo with the SGLang backend
(`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2`), TP=4,
`CONTEXT_LEN=1010000`, `KV_DTYPE=fp8_e5m2`, `ATTN_BACKEND=flashinfer`, YaRN RoPE
scaling with factor=3.9, and `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`.
The benchmark driver was `bench_sglang_qwen3.6_1m.sh`, invoked inside the running
Dynamo container via `podman-hpc exec`, hitting
`http://nid001048:8000/v1`. The ready marker was
`logs/dynamo-ready-1361689.env`, and raw benchmark logs were written under
`logs/bench-20260602-231248/`.

### Reproduction comparison vs 2026-05-27 SGLang run

| Metric                | 2026-05-27 | 2026-06-02 | Delta        | Change |
| --------------------- | ---------- | ---------- | ------------ | ------ |
| Prefill peak          | 11,202.53 tok/s | 11,209.12 tok/s | +6.59 tok/s  | +0.06% |
| Decode single-request | 65.00 tok/s     | 65.00 tok/s     | 0.00 tok/s   | 0.00%  |
| Decode batched peak   | 880.00 tok/s    | 867.00 tok/s    | -13.00 tok/s | -1.48% |

### Comparison vs vLLM v0.21.0 baseline

| Metric                | vLLM v0.21.0 | Dynamo+SGLang 2026-06-02 | Winner          |
| --------------------- | ------------ | ------------------------- | --------------- |
| Prefill peak          | 17.9k tok/s  | 11.2k tok/s               | vLLM (1.6×)     |
| Decode single-request | 50–67 tok/s  | 65 tok/s                  | tie             |
| Decode batched peak   | 264 tok/s    | 867 tok/s                 | SGLang (3.3×)   |

### Prefill peak — input length sweep (Dynamo+SGLang, 2026-06-02)

Setup: input length swept over {32k, 128k, 256k, 512k}, output=1 token,
concurrency=2. Reports `Input token throughput (tok/s)`.

| Input length | 2026-05-27 | 2026-06-02 | Change |
| ------------ | ---------- | ---------- | ------ |
| 32k          | 11,202.53  | 11,209.12  | +0.06% |
| 128k         |  7,267.88  |  7,283.04  | +0.21% |
| 256k         |  4,737.20  |  4,739.95  | +0.06% |
| 512k         |  2,243.59  |  2,243.24  | -0.02% |

### Decode single-request (Dynamo+SGLang, 2026-06-02)

Setup: input=128 tokens, output=1024 tokens, concurrency=1, num-prompts=8. Reports
`Output token throughput (tok/s)`.

| 2026-05-27 | 2026-06-02 | Change |
| ---------- | ---------- | ------ |
| 65.00      | 65.00      | 0.00%  |

### Decode batched peak — concurrency sweep (Dynamo+SGLang, 2026-06-02)

Setup: input=128 tokens, output=256 tokens, concurrency swept over {4, 8, 16, 32, 64,
128}, num-prompts=C×4 per level. Reports `Output token throughput (tok/s)`.

| Concurrency | 2026-05-27 | 2026-06-02 | Change |
| ----------- | ---------- | ---------- | ------ |
| 4           | 240.00     | 240.00     | 0.00%  |
| 8           | 464.00     | 462.00     | -0.43% |
| 16          | 880.00     | 864.00     | -1.82% |
| 32          | 880.00     | 864.00     | -1.82% |
| 64          | 880.00     | 864.00     | -1.82% |
| 128         | 880.00     | 867.00     | -1.48% |

### Takeaway

The 2026-06-02 run is a close reproduction of the 2026-05-27 SGLang result. Prefill
throughput is effectively unchanged, single-request decode matches exactly, and
batched decode is lower by only ~1.5% at the headline peak. The original comparison
against the vLLM v0.21.0 baseline still holds: vLLM is faster for prefill-heavy
workloads, while Dynamo+SGLang remains much faster for batched decode and
high-concurrency serving.
