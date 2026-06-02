# NVIDIA Dynamo on Perlmutter

NVIDIA Dynamo deployed on NERSC Perlmutter, with both supported backends:

- **SGLang** ([`sglang/`](sglang/)) — verified smoke test, multi-node SGLang, a
  Qwen3.6-27B @ 1M-context benchmark stack, and a like-for-like Dynamo+SGLang
  vs. vLLM v0.21.0 comparison.
- **vLLM** ([`vllm/`](vllm/)) — verified smoke test on the
  [`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime?version=1.1.1)
  image, single- and multi-node (torch.distributed) launchers.

Both launchers allocate GPU nodes through Slurm, start the Dynamo frontend on
the head node, bring up one worker per allocated node via `srun` + `podman-hpc`,
wait for worker registration through the Dynamo `/health` endpoint, and save a
test chat-completion response to `logs/`. Discovery uses the file backend
(`--discovery-backend file`) with explicit `--request-plane tcp --event-plane zmq`
so no NATS/etcd infrastructure is needed for a single-job run.

## Prerequisites

One-time setup on Perlmutter before the first submit.

**Useful links**

- NVIDIA Dynamo on NGC (catalog search): <https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=dynamo>
  — [`sglang-runtime`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime),
  [`vllm-runtime`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime)
- NVIDIA Dynamo source: <https://github.com/ai-dynamo/dynamo>
- NVIDIA Dynamo docs (vLLM backend reference): <https://github.com/ai-dynamo/dynamo/tree/main/docs/backends/vllm>
- `podman-hpc` on NERSC: <https://docs.nersc.gov/development/containers/podman-hpc/overview/>
- Hugging Face access tokens: <https://huggingface.co/settings/tokens>

### 1. Authenticate with NGC Registry

Get an NGC API key at <https://ngc.nvidia.com> → Setup → Generate API Key, then:

```bash
podman-hpc login nvcr.io
```

When prompted:

```text
Username: $oauthtoken
Password: <your NGC API key>
```

### 2. Pull and migrate both runtime images

```bash
# SGLang backend
podman-hpc pull    nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
podman-hpc migrate nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1

# vLLM backend
podman-hpc pull    nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
podman-hpc migrate nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
```

The `migrate` step squashes each image into PSCRATCH so subsequent jobs avoid
the home-quota and rebuild overhead.

### 3. (Optional) Set a Hugging Face token

Only required for gated/private/license-gated models (Llama, Gemma, etc.); not
needed for the default `Qwen/Qwen3-0.6B`. Get a token at
<https://huggingface.co/settings/tokens>, then either:

```bash
export HF_TOKEN=hf_your_token_here
```

or save it to `~/.hf_token` (chmod 600) — every launcher reads either.

### 4. Pre-download Model to `$SCRATCH`

The launchers set `HF_HOME=$SCRATCH/cache` and mount `$SCRATCH` into the
container, so anything cached at that path is picked up automatically.
Pre-downloading on a login node avoids burning GPU walltime on the initial
fetch.

Download:

```bash
# On a login node (no GPU needed)
module load python                           # or activate your env
pip install --user -U "huggingface_hub[cli]" # provides the new `hf` CLI

export HF_HOME=$SCRATCH/cache
mkdir -p "$HF_HOME"

# Gated models need a token:
export HF_TOKEN=$(cat ~/.hf_token)
# or interactive: hf auth login

hf download Qwen/Qwen3.6-27B --repo-type model --cache-dir "$HF_HOME"

# Optional: Nemotron Ultra 253B
hf download nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 --repo-type model --cache-dir "$HF_HOME"
```

Note: `huggingface-cli` was renamed to `hf`. Old `huggingface-cli` commands
print a deprecation warning and no longer execute.

Check availability for any downloaded model:

```bash
MODEL_ID=Qwen/Qwen3.6-27B
# MODEL_ID=nvidia/Llama-3_1-Nemotron-Ultra-253B-v1

export MODEL_CACHE="$HF_HOME/models--${MODEL_ID//\//--}"

# Where it lives + total size on disk
du -sh "$MODEL_CACHE"

# Per-shard sizes (resolved through symlinks -> real files)
ls -lhL "$MODEL_CACHE"/snapshots/*/*.safetensors

# Integrity checks: no zero-byte blobs, all referenced shards present
find "$MODEL_CACHE"/blobs -size 0
python3 -c "import glob, json, os; idx=glob.glob(os.environ['MODEL_CACHE'] + '/snapshots/*/model.safetensors.index.json')[0]; d=json.load(open(idx)); print(len(set(d['weight_map'].values())), 'unique shards')"
```

### 5. Edit the Slurm account

Replace `<YOUR ACCOUNT>` in the `#SBATCH -A` line of [`launch_dynamo.sh`](launch_dynamo.sh)
(and any backend-specific launcher you plan to run) with your NERSC project
charge code.

## Quickstart

Use the unified launcher [`launch_dynamo.sh`](launch_dynamo.sh) and pick the backend
with `BACKEND=`:

```bash
mkdir -p logs

# Defaults: BACKEND=sglang, 1 node, 4 GPUs, TP=4, Qwen/Qwen3-0.6B, ctx=4096
sbatch launch_dynamo.sh

# Same on vLLM
BACKEND=vllm sbatch launch_dynamo.sh

# Qwen3.6-27B, either backend (MAX_MODEL_LEN caps the KV alloc on both)
MODEL_NAME=Qwen/Qwen3.6-27B MAX_MODEL_LEN=8192 sbatch launch_dynamo.sh
MODEL_NAME=Qwen/Qwen3.6-27B MAX_MODEL_LEN=8192 BACKEND=vllm sbatch launch_dynamo.sh

# Nemotron Ultra 253B, either backend (adjust MAX_MODEL_LEN for the target run)
MODEL_NAME=nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 MAX_MODEL_LEN=8192 sbatch launch_dynamo.sh
MODEL_NAME=nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 MAX_MODEL_LEN=8192 BACKEND=vllm sbatch launch_dynamo.sh

# Gated model (needs HF_TOKEN), vLLM, 2 nodes
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct BACKEND=vllm \
    sbatch --nodes=2 launch_dynamo.sh
```

| Env var | Default | What it controls |
| --- | --- | --- |
| `BACKEND` | `sglang` | `sglang` or `vllm` |
| `MODEL_NAME` | `Qwen/Qwen3-0.6B` | Any HF model id; pre-download to `$SCRATCH/cache` to avoid in-job fetch |
| `MAX_MODEL_LEN` | `4096` | Context cap; required for models whose native ctx exceeds GPU memory (e.g. Qwen3.6-27B native=256k → use 8192) |
| `MAX_NUM_SEQS` | `8` | Max concurrent sequences (vLLM `--max-num-seqs`; ignored by SGLang) |
| `HF_TOKEN` | from `~/.hf_token` | Required for gated models (Llama, Gemma, …) |

The backend-specific smoke launchers ([`sglang/launch_sglang_smoke.sh`](sglang/launch_sglang_smoke.sh),
[`vllm/launch_vllm_smoke.sh`](vllm/launch_vllm_smoke.sh)) are still checked in as
single-backend references; the unified script is the recommended entry point.

Then check the response:

```bash
python3 -m json.tool logs/dynamo-response-<job-id>.json
```

## Layout

Top-level:

- [`README.md`](README.md) — this file
- [`launch_dynamo.sh`](launch_dynamo.sh) — **unified launcher**, dispatches on `BACKEND=sglang|vllm` (default `sglang`)
- [`main.py`](main.py), [`pyproject.toml`](pyproject.toml) — Python harness for OpenAI-client tests

SGLang backend ([`sglang/`](sglang/), image v1.1.1):

| File | What it is |
| --- | --- |
| [`sglang/README_sglang.md`](sglang/README_sglang.md) | `sbatch` quickstart |
| [`sglang/sglang_interactive.md`](sglang/sglang_interactive.md) | Interactive (manual `podman-hpc`) walkthrough |
| [`sglang/sglang-vs-vllm.md`](sglang/sglang-vs-vllm.md) | Qwen3.6-27B @ 1M ctx: Dynamo+SGLang vs vLLM v0.21.0 |
| [`sglang/launch_sglang_smoke.sh`](sglang/launch_sglang_smoke.sh) | 1- or N-node smoke launcher |
| [`sglang/launch_sglang_qwen3.6-1m.sh`](sglang/launch_sglang_qwen3.6-1m.sh) | 1M-ctx benchmark launcher (TP=4, fp8 KV, YaRN) |
| [`sglang/launch_sglang_standalone.sh`](sglang/launch_sglang_standalone.sh) | Standalone SGLang (no Dynamo) for comparison |

vLLM backend ([`vllm/`](vllm/), image v1.1.1):

| File | What it is |
| --- | --- |
| [`vllm/README_vllm.md`](vllm/README_vllm.md) | `sbatch` quickstart |
| [`vllm/launch_vllm_smoke.sh`](vllm/launch_vllm_smoke.sh) | 1- or N-node smoke launcher |

## Where to read next

| Topic | Doc |
| --- | --- |
| SGLang setup, smoke test, multi-node | [`sglang/README_sglang.md`](sglang/README_sglang.md) |
| SGLang interactive (manual) workflow | [`sglang/sglang_interactive.md`](sglang/sglang_interactive.md) |
| Qwen3.6-27B @ 1M ctx: SGLang vs vLLM | [`sglang/sglang-vs-vllm.md`](sglang/sglang-vs-vllm.md) |
| vLLM backend (v1.1.1) quickstart | [`vllm/README_vllm.md`](vllm/README_vllm.md) |

## Standalone SGLang (no Dynamo)

For an apples-to-apples comparison with the Dynamo-managed path, the repo also
ships a plain SGLang launcher: [`sglang/launch_sglang_standalone.sh`](sglang/launch_sglang_standalone.sh).

```bash
sbatch sglang/launch_sglang_standalone.sh
```
