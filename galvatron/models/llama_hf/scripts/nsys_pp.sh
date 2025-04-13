export PATH=/opt/nvidia/nsight-systems/2025.2.1/bin:$PATH
nsys profile \
    -w true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s cpu \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cudabacktrace=true \
    -x true \
    --force-overwrite true \
    -o ./nsys/galvatron-$(date +"%Y%m%d_%H%M%S") \
    bash ./scripts/pp_train.sh