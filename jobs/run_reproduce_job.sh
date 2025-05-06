#!/bin/bash

usage() {
    echo "Usage: $0 --openai-key <OPENAI_API_KEY> --wandb-key <WANDB_API_KEY> --job-script <JOB_SCRIPT>"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --openai-key) OPENAI_API_KEY="$2"; shift ;;
        --wandb-key) WANDB_API_KEY="$2"; shift ;;
        --job-script) JOB_SCRIPT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "$OPENAI_API_KEY" ] || [ -z "$WANDB_API_KEY" ] || [ -z "$JOB_SCRIPT" ]; then
    echo "Error: Missing required arguments."
    usage
fi

