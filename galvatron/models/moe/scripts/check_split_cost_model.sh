export NUM_NODES=1
export NUM_GPUS_PER_NODE=8

MODEL_SIZE="mixtral-8x7b"
SEQ=2048
IS_MOE=1

MODEL_ARGS="
    --model_size ${MODEL_SIZE} \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --set_seqlen_manually 1 \
    --set_experts_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 4 \
    --num_attention_heads 32 \
    --seq_length ${SEQ}"

COST_MODEL_ARGS="
    ${MODEL_ARGS} \
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --mixed_precision bf16 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --sequence_parallel \
    --time_profile_mode batch batch \
    --memory_profile_mode static static \
    --profile_granularity split \
    --is_MoE ${IS_MOE} \
    --num_experts 8 \
    --top_k 2 \
"

VERSION_OPTION_ARGS="
    --zero_with_slight_noise 1 \
    --estimate_tp_time_type fit \
"

python3 check_split_cost_model.py ${COST_MODEL_ARGS} ${VERSION_OPTION_ARGS}