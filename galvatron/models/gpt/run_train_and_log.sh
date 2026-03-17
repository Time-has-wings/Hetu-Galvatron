#!/bin/bash
# Run train_yaml.sh and capture all output to run_output.txt
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../.. && pwd)"
export NPROC_PER_NODE=2
exec bash scripts/train_yaml.sh 2>&1 | tee run_output.txt
