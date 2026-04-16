set -x
set -o pipefail

log_dir="logs/search_engine"
mkdir -p $log_dir

python3 search_dist.py scripts/search_dist.yaml 2>&1 | tee $log_dir/search_engine.log
