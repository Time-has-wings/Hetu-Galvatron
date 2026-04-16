set -x
set -o pipefail

log_dir="logs/profile_computation"
mkdir -p $log_dir

export RUNTIME_LAUNCHER="torchrun --nnodes 1 --nproc_per_node 1 train_dist.py "
python3 profiler.py scripts/profile_computation.yaml 2>&1 | tee $log_dir/profile_computation.log
