

NVIDIA Dynamo, a distributed inference serving framework, was successfully deployed on NERSC Perlmutter using the SGLang backend. The official NVIDIA Dynamo container image (`sglang-runtime:1.0.2`) was pulled and migrated to `$PSCRATCH` to manage storage efficiently. A GPU compute node on Perlmutter was allocated through Slurm, and the container was launched with `podman-hpc`. Inside the container, the OpenAI-compatible frontend was started on port 8000, and the Qwen3-0.6B language model was deployed as a single-GPU worker. The deployment was verified with a health check and a chat completion request to the endpoint, which returned a valid model response. This confirms that NVIDIA Dynamo is functional on Perlmutter for the single-GPU SGLang path and ready for further experimentation with larger models and more advanced deployment examples such as tensor parallelism, disaggregated serving, and multi-node inference.

# NVIDIA Dynamo Quickstart on Perlmutter (SGLang Backend)

## Prerequisites
- Access to NERSC Perlmutter
- NGC account with API key
- HuggingFace account (for gated models)

---

## Step 1 — Authenticate with NGC Registry

```bash
podman-hpc login nvcr.io
```

When prompted:
```
Username: $oauthtoken
Password: <your NGC API key>
```

> Get your NGC API key at: https://ngc.nvidia.com → Setup → Generate API Key

---

## Step 2 — Pull and Migrate Container Image to PSCRATCH

Pull the image and migrate to `$PSCRATCH` to avoid home directory quota issues:

```bash
podman-hpc pull nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2
podman-hpc migrate nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2
```

---

## Step 3 — Request a Compute Node

### CPU only (no GPU)
```bash
salloc --nodes=1 --ntasks=1 --account=<your_account> -C cpu -q interactive -t 01:00:00
```

### GPU (recommended, 1 GPU)
```bash
salloc --nodes=1 --ntasks=1 --gpus=1 --account=<your_account> -C gpu -q interactive -t 01:00:00
```

### GPU (4 GPUs for tensor parallelism)
```bash
salloc --nodes=1 --ntasks=1 --gpus=4 --account=<your_account> -C gpu -q interactive -t 01:00:00
```

Confirm GPUs are available (GPU case only):
```bash
nvidia-smi
```

> Replace `<your_account>` with your Perlmutter project account (e.g. `m1234`).
> You can find it by running `groups` on the login node.

---

## Step 4 — Run the Container

### CPU only
```bash
podman-hpc run --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2
```

### GPU
```bash
podman-hpc run \
    --gpu \
    --nccl-cu12 \
    --network host \
    --ipc=host \
    --ulimit memlock=-1 \
    --rm -it \
    nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.2
```

---

## Step 5 — Start the Frontend

Inside the container in Terminal 1:

```bash
python3 -m dynamo.frontend --discovery-backend file
```

Wait for the frontend to fully initialize before proceeding.

> Note: When using a GPU allocation, open any additional terminals by SSHing into the same compute node first, for example `ssh <nodename>`. Use the node name shown in the Slurm allocation prompt, such as `nid001197`.

---

## Step 6 — Attach and Start the Worker

### Terminal 2 — Attach to the running container
```bash
podman-hpc exec -it $(podman-hpc ps -q) bash
```

### CPU only
```bash
python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B \
    --discovery-backend file
```

### GPU (single GPU)
```bash
python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B \
    --discovery-backend file \
    --tensor-parallel-size 1
```

### GPU (4 GPUs with tensor parallelism)
```bash
export NCCL_DEBUG=INFO
export NCCL_CUMEM_HOST_ENABLE=0

python3 -m dynamo.sglang \
    --model-path Qwen/Qwen3-0.6B \
    --discovery-backend file \
    --tensor-parallel-size 4
```

> For 4-GPU tensor parallelism, the container should be started with `--gpu --nccl-cu12`, host IPC, and unlocked pinned memory. Do not combine `--ipc=host` with `--shm-size`; Podman rejects that combination because host IPC already uses the host shared-memory namespace. If NCCL still fails during initialization, keep `NCCL_DEBUG=INFO` enabled and retry once with `export NCCL_IB_DISABLE=1` to force node-local GPU communication while debugging.

Wait for the model to fully load before proceeding.

 Note: You need a Hugging Face token when the model is not publicly downloadable without authentication.
  >
  > Common cases include:
  > - Gated models, such as Llama, Gemma, some Mistral, Kimi, Qwen-VL, medical/domain models, etc.
  > - Private models in your own Hugging Face account or organization.
  > - License-accepted models where you must log in and accept the model terms on the Hugging Face model page first.
  > - Rate/access issues where authenticated downloads are more reliable.
  >
  > For the current quickstart model, `Qwen/Qwen3-0.6B`, you usually do **not** need a Hugging Face token because the model is public.
  >
  > If you do need a token, set it inside the container in Terminal 2 before starting the worker:
  >
  > ```bash
  > export HF_TOKEN=hf_your_token_here
  > ```
  >
  > Then start the worker:
  >
  > ```bash
  > python3 -m dynamo.sglang \
  >     --model-path <model_name> \
  >     --discovery-backend file \
  >     --tensor-parallel-size 1
  > ```
  >
  > For gated models, first visit the model page on Hugging Face in your browser, accept the license/terms, then use a token from that same Hugging Face account.

---

## Step 7 — Verify and Test

### Terminal 3 — Health check
```bash
curl -sf http://localhost:8000/health && echo OK
```

Expected output (worker connected):
```json
{"status":"healthy","endpoints":["dyn://dynamo.backend.generate"],"instances":[{...}]}OK
```

Expected output (no worker yet):
```json
{"status":"healthy","endpoints":[],"instances":[]}OK
```

> If instances is empty, the worker is still loading — wait and retry.

### Send a chat request
```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "What is 2+2?"}],
       "max_tokens": 256}' | python3 -m json.tool
```

### Expected Response
```json
{
    "id": "chatcmpl-...",
    "choices": [
        {
            "index": 0,
            "message": {
                "content": "...",
                "role": "assistant"
            },
            "finish_reason": "stop"
        }
    ],
    "model": "Qwen/Qwen3-0.6B",
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": "...",
        "total_tokens": "..."
    }
}
```

> If the response still ends with `"finish_reason": "length"`, increase `max_tokens` further, for example to `512`.

---

## Notes
- The username for NGC login is literally `$oauthtoken`
- `--discovery-backend file` avoids needing etcd/nats for simple cases
- `--tensor-parallel-size` must not exceed the number of available GPUs
- Qwen3-0.6B is a small model and works well on 1 GPU, but it can still be used as a smoke test for 4-GPU tensor parallelism
- For gated models (Llama, Kimi, Qwen-VL), set `export HF_TOKEN=hf_...` before launching
- If `$(podman-hpc ps -q)` returns multiple IDs, replace with the specific container ID from `podman-hpc ps`

---

## References
- [NVIDIA Dynamo Quickstart](https://docs.nvidia.com/dynamo/dev/getting-started/quickstart)
- [NVIDIA Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [NERSC Podman-hpc documentation](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
- [NVIDIA NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
