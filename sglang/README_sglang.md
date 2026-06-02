# NVIDIA Dynamo Quickstart on Perlmutter (SGLang Backend)

NVIDIA Dynamo was tested on NERSC Perlmutter using the SGLang backend through the non-interactive Slurm launcher `launch_sglang_smoke.sh`. The script allocates GPU node(s) with Slurm, starts the Dynamo frontend on the head node, and launches one Dynamo SGLang worker per allocated node using `podman-hpc` and the migrated `sglang-runtime:1.1.1` container image from `$PSCRATCH`. The deployment was validated with the `Qwen/Qwen3-0.6B` model using the Dynamo `/health` endpoint and an OpenAI-compatible chat completion request. 

## Prerequisites

- Access to NERSC Perlmutter
- NGC account with API key
- Hugging Face account/token for gated models

## Step 1 - Authenticate with NGC Registry

```bash
podman-hpc login nvcr.io
```

When prompted:

```text
Username: $oauthtoken
Password: <your NGC API key>
```

Get your NGC API key at: <https://ngc.nvidia.com> -> Setup -> Generate API Key.

## Step 2 - Pull and Migrate Container Image to PSCRATCH

Pull the image and migrate it to `$PSCRATCH` to avoid home directory quota issues:

```bash
podman-hpc pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
podman-hpc migrate nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
```

The batch script uses:

```bash
nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
```

## Step 3 - Optional Hugging Face Token

You do not need a Hugging Face token for the default public quickstart model:

```bash
Qwen/Qwen3-0.6B
```

You do need a Hugging Face token for gated, private, or license-accepted models, for example:

```bash
meta-llama/Llama-3.1-8B-Instruct
```

If the model needs Hugging Face authentication, either export `HF_TOKEN` before submitting the job or save a token where the batch script can read it.

Option 1, use an environment variable for the current shell:

```bash
export HF_TOKEN=hf_your_token_here
```

Option 2, save it in your home directory:

```bash
echo "$HF_TOKEN" > "$HOME/.hf_token"
chmod 600 "$HOME/.hf_token"
```
You usually need a Hugging Face token for:

- gated models such as Llama, Gemma, some Mistral, Kimi, Qwen-VL, medical/domain models, and similar
- private models in your Hugging Face account or organization
- license-accepted models where you must accept terms on the Hugging Face model page first
- authenticated downloads when rate limits or access checks are more reliable with a token

## Step 4 - Submit the Batch Job

From this directory:

```bash
mkdir -p logs
sbatch launch_sglang_smoke.sh
```

To test more nodes without editing the script:

```bash
sbatch --nodes=2 launch_sglang_smoke.sh
sbatch --nodes=3 launch_sglang_smoke.sh
```

To test a gated Hugging Face model, first make sure your Hugging Face account has accepted the model license, then submit with `MODEL_NAME`:

```bash
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct sbatch launch_sglang_smoke.sh
```

For that model, the launcher will use your `HF_TOKEN` from the environment or `$HOME/.hf_token`. Gated or larger models can take longer to download and load, and may fail if your token does not have access.

To test a small public Hugging Face model without changing the script, use:

```bash
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 sbatch launch_sglang_smoke.sh
```

This path was tested successfully with the included smoke test.

Smoke-test models verified so far:

| Model | Nodes | GPUs | TP | Submit command (example) |
| --- | ---: | ---: | ---: | --- |
| `Qwen/Qwen3-0.6B` | 1 | 4 | 4 | `sbatch launch_sglang_smoke.sh` |
| `Qwen/Qwen3-0.6B` | 2 | 8 | 8 | `sbatch --nodes=2 launch_sglang_smoke.sh` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1 | 4 | 4 | `MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 sbatch launch_sglang_smoke.sh` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2 | 8 | 8 | `MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 sbatch --nodes=2 launch_sglang_smoke.sh` |

You can also edit `#SBATCH --nodes` for 1 node, 2 nodes, 3 nodes, or more directly and edit `TP` in `launch_sglang_smoke.sh` if your allocation or model requires it. You can set `MODEL_NAME` at submit time without editing the script. By default, `TP` is computed as `4 * SLURM_JOB_NUM_NODES`, matching 4 GPUs per allocated node.

## What the Job Does

The batch script starts the Dynamo frontend on the head node:

```bash
python3 -m dynamo.frontend --discovery-backend file
```

For a single node, it starts one SGLang worker:

```bash
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --tensor-parallel-size 4
```

For multiple nodes, it starts one worker per node with SGLang distributed flags:

```bash
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --tensor-parallel-size <total-gpus> \
  --dist-init-addr <head-node>:50000 \
  --nnodes <node-count> \
  --node-rank <rank>
```

For example, a 2-node job with 4 GPUs per node uses:

```text
--tensor-parallel-size 8
--nnodes 2
--node-rank 0   # first node
--node-rank 1   # second node
```

The job waits for the worker to register, then sends this OpenAI-compatible chat completion request:

```text
Explain what NVIDIA Dynamo does on top of SGLang.
```


## Outputs

Slurm output goes to:

```bash
logs/launch_sglang_smoke.sh-<job-id>.out
logs/launch_sglang_smoke.sh-<job-id>.err
```

By default, `.out` contains only high-level job information and `.err` should usually be empty. Frontend and worker runtime logs are redirected to `/dev/null` to keep the log directory small. The model response goes to:

```bash
logs/dynamo-response-<job-id>.json
```

Check the response with:

```bash
python3 -m json.tool "logs/dynamo-response-<job-id>.json"
```


## Notes

- This job exits after the smoke-test request succeeds.
- The 1-node and 2-node paths have been run and verified with the included smoke test.
- The `MODEL_NAME` override was tested with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on 1 node and 2 nodes.
- The multi-node path is implemented in the same launcher using Slurm's node list and SGLang distributed arguments.
- `NCCL_DEBUG` is intentionally unset by default to reduce log noise.
- `--discovery-backend file` uses a per-job shared discovery directory mounted into the containers.
- `TP` should match the number of GPUs requested across all Slurm nodes.
