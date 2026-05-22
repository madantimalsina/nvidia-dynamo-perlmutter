# NVIDIA Dynamo Quickstart on Perlmutter (SGLang Backend)

NVIDIA Dynamo has been deployed on NERSC Perlmutter using the SGLang backend through the Slurm launcher `launch_dynamo.sh`. This launcher allocates GPU nodes with Slurm, starts the Dynamo frontend on the head node, and launches one Dynamo SGLang worker per allocated node using `srun`, `podman-hpc`, GPU/NCCL support, and the official NVIDIA Dynamo SGLang runtime container. It dynamically derives the Slurm node list, supports single-node and multi-node runs, waits for worker registration through the Dynamo health endpoint, and saves a test chat-completion response to the `logs/` directory.

The repository also includes an interactive quickstart in [NvidiaDynamo.md](NvidiaDynamo.md), showing the manual workflow for authenticating with NGC, pulling and migrating the NVIDIA Dynamo SGLang runtime image, requesting a Perlmutter GPU allocation, launching the container with `podman-hpc`, starting the Dynamo frontend and SGLang worker, and verifying the OpenAI-compatible endpoint.

In addition to the NVIDIA Dynamo workflow, the repository includes a standalone SGLang deployment script, `launch.sh`, for running SGLang directly without NVIDIA Dynamo. This provides a useful comparison path between plain SGLang serving and Dynamo-managed SGLang serving on Perlmutter.


## Getting started

Make sure to make the logs directory for Slurm logs. 

```bash
mkdir $PWD/logs
```

## NVIDIA-DYNAMO

NVIDIA Dynamo deployment notes are available in two versions:

- [NvidiaDynamo.md](NvidiaDynamo.md): non-interactive `sbatch` workflow using `launch_dynamo.sh`
- [NvidiaDynamo_interactive.md](NvidiaDynamo_interactive.md): for interactive workflow

For the tested non-interactive path, start with:

```bash
sbatch launch_dynamo.sh
```

## Setting up hugging face token

For some models you will need a hugging face token. The launcher script can
read that token from a file in your home directory.

```bash
export HF_TOKEN=hf_your_token_here
```

and/or, save it in your home directory:

```bash
echo "$HF_TOKEN" > "$HOME/.hf_token"
chmod 600 "$HOME/.hf_token"
```

Make sure to make the logs directory for Slurm logs. 
# sglang-deployment (without NVIDIA-DYNAMO)
```bash
mkdir $PWD/logs

sbatch launch.sh
```
