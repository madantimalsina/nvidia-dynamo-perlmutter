# sglang-deployment



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