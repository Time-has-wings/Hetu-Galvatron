set -x
set -o pipefail

export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

log_dir="logs/profile_hardware"
mkdir -p $log_dir

python3 profile_hardware.py scripts/profile_hardware.yaml 2>&1 | tee $log_dir/profile_hardware.log
