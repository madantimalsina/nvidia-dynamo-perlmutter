# NVIDIA Dynamo Quickstart on Perlmutter (vLLM Backend, v1.1.1)

Companion to [`../sglang/README_sglang.md`](../sglang/README_sglang.md). Same Slurm-driven flow, same
`Qwen/Qwen3-0.6B` smoke test, same OpenAI-compatible chat completion sanity check —
but with the Dynamo **vLLM** worker (`dynamo.vllm`) on the latest NGC image
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1`. Launcher:
[`launch_vllm_smoke.sh`](launch_vllm_smoke.sh).

The flag set follows upstream Dynamo's reference
[`examples/backends/vllm/launch/agg.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg.sh)
(single-node) and
[`multi_node_tp.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/multi_node_tp.sh)
(multi-node TP).

## Differences from the SGLang launcher

| Piece | SGLang (`launch_sglang_smoke.sh`) | vLLM v1.1.1 (`launch_vllm_smoke.sh`) |
| --- | --- | --- |
| Container image | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1` | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` |
| Worker module | `python3 -m dynamo.sglang` | `python3 -m dynamo.vllm` |
| Model flag | `--model-path` | `--model` |
| Context length | `--context-length` | `--max-model-len` |
| Concurrent seqs | `--max-running-requests` | `--max-num-seqs` |
| KV dtype | `--kv-cache-dtype fp8_e5m2` | `--kv-cache-dtype fp8` |
| Multi-node TP | `--dist-init-addr <head>:50000 --nnodes N --node-rank R` | `--nnodes N --node-rank R --master-addr <head> --master-port 29500` (+ `--headless` on rank > 0) |
| Discovery | `--discovery-backend file` | `--discovery-backend file` (same; uses `DYN_FILE_KV`) |
| Quick-start tweaks | none | `--enforce-eager` (skips CUDA graphs for faster start) |

The vLLM backend uses **torch.distributed** for multi-node, not Ray, despite what
some external blog posts claim. The Dynamo `multi_node_tp.sh` reference script
makes this explicit.

## Prerequisites

- Access to NERSC Perlmutter
- NGC account with API key
- Hugging Face account/token only for gated models (not needed for `Qwen/Qwen3-0.6B`)

## Step 1 — Authenticate with NGC Registry

```bash
podman-hpc login nvcr.io
```

Username: `$oauthtoken`, Password: your NGC API key
(<https://ngc.nvidia.com> → Setup → Generate API Key).

## Step 2 — Pull and Migrate the vLLM 1.1.1 Container to PSCRATCH

```bash
podman-hpc pull    nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
podman-hpc migrate nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
```

Catalog page:
<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.1.1>.

## Step 3 — Optional Hugging Face Token

Not needed for `Qwen/Qwen3-0.6B`. For gated models, export `HF_TOKEN` or save it
to `~/.hf_token` (chmod 600) — the launcher reads either.

## Step 4 — Submit the Batch Job

```bash
mkdir -p logs
sbatch launch_vllm_smoke.sh                                          # 1 node,  TP=4
sbatch --nodes=2 launch_vllm_smoke.sh                                # 2 nodes, TP=8
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 sbatch launch_vllm_smoke.sh
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct  sbatch launch_vllm_smoke.sh   # gated
```

`TP` is computed as `4 * SLURM_JOB_NUM_NODES` (4 GPUs per node).

Tunables you can override at submit time:

```bash
MAX_MODEL_LEN=8192 MAX_NUM_SEQS=16 sbatch launch_vllm_smoke.sh
MASTER_PORT=29501 sbatch --nodes=2 launch_vllm_smoke.sh
```

## What the Job Does

1. Start the Dynamo frontend on the head node:

   ```bash
   python3 -m dynamo.frontend --discovery-backend file
   ```

2. Start one `dynamo.vllm` worker per node via `srun`. The head worker
   (`--node-rank 0`) registers the OpenAI endpoint; secondary nodes are started
   with `--headless` (vLLM worker only, no Dynamo endpoint of their own):

   ```bash
   # 1 node
   python3 -m dynamo.vllm \
       --model Qwen/Qwen3-0.6B \
       --discovery-backend file \
       --tensor-parallel-size 4 \
       --max-model-len 4096 \
       --max-num-seqs 8 \
       --enforce-eager

   # 2 nodes, head (rank 0)
   python3 -m dynamo.vllm \
       --model Qwen/Qwen3-0.6B \
       --discovery-backend file \
       --tensor-parallel-size 8 \
       --max-model-len 4096 --max-num-seqs 8 --enforce-eager \
       --nnodes 2 --node-rank 0 --master-addr <head> --master-port 29500

   # 2 nodes, secondary (rank 1)
   python3 -m dynamo.vllm \
       --model Qwen/Qwen3-0.6B \
       --discovery-backend file \
       --tensor-parallel-size 8 \
       --max-model-len 4096 --max-num-seqs 8 --enforce-eager \
       --nnodes 2 --node-rank 1 --master-addr <head> --master-port 29500 \
       --headless
   ```

3. Wait for `/health` to report at least one registered instance, then POST the
   smoke-test prompt (`Explain what NVIDIA Dynamo does on top of vLLM.`) to
   `/v1/chat/completions`.

## Outputs

```bash
logs/launch_vllm_smoke.sh-<job-id>.out      # high-level launcher log
logs/launch_vllm_smoke.sh-<job-id>.err
logs/dynamo-frontend-<job-id>.log            # frontend stdout/stderr
logs/dynamo-worker-<job-id>-<rank>-<node>.log
logs/dynamo-response-<job-id>.json           # parsed chat completion response
```

Check the response:

```bash
python3 -m json.tool logs/dynamo-response-<job-id>.json
```

## Notes

- The 1-node path is the smallest possible smoke test — submit that first to
  validate the image + frontend + worker plumbing before trying `--nodes=2`.
- The launcher passes `DYN_DISCOVERY_BACKEND=file` and
  `DYN_FILE_KV=/tmp/dynamo_store_kv` into every container, with the host path
  `$SCRATCH/dynamo_store_kv_<jobid>` bind-mounted in. This avoids the NATS/etcd
  infrastructure that upstream's `docker-compose.yml` brings up — fine for
  smoke testing one job at a time.
- `--enforce-eager` skips vLLM's CUDA-graph capture step. It speeds startup
  dramatically (relevant for a smoke test) at the cost of decode throughput.
  Drop it for a real perf measurement.
- `TP` must equal the total GPU count across all allocated nodes.
- For multi-node, secondary nodes use `--headless`; only the head rank registers
  the Dynamo OpenAI endpoint. The `--master-addr/--master-port` pair bootstraps
  vLLM's torch.distributed group.
- The job exits after the smoke-test request returns HTTP 200; the cleanup trap
  stops the worker and frontend on the way out.
