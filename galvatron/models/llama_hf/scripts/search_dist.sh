export NUM_NODES=1
export NUM_GPUS_PER_NODE=8

MODEL_SIZE="llama-7b"
MEMORY=36
SEQ=2048
FINE_GRAINED=1

# =====================cost model args==================
MODEL_ARGS="
    --model_size ${MODEL_SIZE} \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 1 \
    --set_experts_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 24 \
    --num_attention_heads 32 \
    --seq_length ${SEQ}"

CLUSTER_INFO="
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
"

TRAIN_ARGS="
    --mixed_precision bf16 \
    --pipeline_type pipedream_flush \
    --sequence_parallel \
    --async_grad_reduce 1 \
"

PROFILE_MODEL_ARGS="
    --time_profile_mode batch \
    --memory_profile_mode static \
    --profile_granularity together \
"

VERSION_OPTION_AGRS="
    --zero_with_slight_noise 1 \
    --estimate_tp_time_type fit \
"

COST_MODEL_ARGS="
    ${MODEL_ARGS} \
    ${CLUSTER_INFO} \
    ${TRAIN_ARGS} \
    ${PROFILE_MODEL_ARGS} \
    ${VERSION_OPTION_AGRS} \
"

# ==============search engine args===============
SYSTEM_INFO_ARGS="
    --memory_constraint ${MEMORY} \
    --fine_grained_mode ${FINE_GRAINED} \
    --parallel_search \
"
#     --parallel_search \

BSZ_ARGS="
    --min_bsz 16 \
    --max_bsz 16 \
    --bsz_scale 1 \
    --settle_bsz -1 \
    --recommend_min_bsz 0 \
"

STRATEGY_ARGS="
    --disable_pp 0 \
    --disable_tp 0 \
    --disable_sp 0 \
    --disable_cp 1 \
    --disable_dp 0 \
    --disable_ckpt 0 \
    --disable_fsdp 0 \
    --disable_embedding_lmhead_tp 0 \
    --disable_embedding_lmhead_sp 0 \
    --disable_tp_consec 1 \
    --max_tp_deg 8 \
    --max_pp_deg 8 \
    --max_sp_deg 8 \
    --max_cp_deg 8 \
"

TRAIN_CONFIG_ARGS="
    --default_dp_type zero2 \
"

SEARCH_ENGINE_ARGS="
    ${SYSTEM_INFO_ARGS} \
    ${BSZ_ARGS} \
    ${STRATEGY_ARGS} \
    ${TRAIN_CONFIG_ARGS} \
"

# =======================run=====================
BACKGROUND=0

if [ $BACKGROUND -eq 1 ]; then
    echo "Search in background..."
    if [ ! -d "log" ]; then
        mkdir -p "log"
    fi
    OUTPUT_FILE="log/Search_${MODEL_SIZE}_${MEMORY}GB_${NUM_NODES}Nodes_${NUM_GPUS_PER_NODE}GPUs_per_node_${SEQ}_${FINE_GRAINED}.log"
    nohup python3 search_dist.py ${COST_MODEL_ARGS} ${SEARCH_ENGINE_ARGS} 1> ${OUTPUT_FILE} 2>&1 &
else
    echo "Search in foreground..."
    python3 search_dist.py ${COST_MODEL_ARGS} ${SEARCH_ENGINE_ARGS}
fi