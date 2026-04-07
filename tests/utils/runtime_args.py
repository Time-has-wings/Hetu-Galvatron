"""Test-only argument namespace that mirrors GalvatronRuntimeArgs structure.

Provides the nested ``args.parallel.*``, ``args.model.*``, ``args.train.*``,
``args.profile.*``, ``args.ckpt.*``, ``args.data.*`` interface expected by the
Galvatron core runtime, while remaining a plain Python object for easy
mutation in test code.
"""
import torch


class _Namespace:
    """Minimal attribute bag — allows arbitrary ``setattr``."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_test_args(
    hf_arch="gpt",
    rank=0,
    world_size=1,
    checkpoint_load=None,
    mixed_precision="fp32",
    async_grad_reduce=True,
    galvatron_config_path=None,
    global_batch_size=16,
    chunks=2,
    seed=42,
    seq_length=32,
    hidden_size=128,
    num_layers=4,
    num_attention_heads=4,
    ffn_hidden_size=512,
    vocab_size=1000,
    use_flash_attn=True,
    sequence_parallel=True,
    use_ulysses=False,
    model_size=None,
    group_query_attention=False,
    num_query_groups=None,
    norm_epsilon=1e-5,
):
    """Build a test args namespace compatible with the Galvatron runtime.

    ``args.model.model_type`` is always ``\"gpt\"`` (same decoder stack in the runtime).
    ``hf_arch`` selects HF weight layout / module details: ``\"gpt\"`` (GPT-2), ``\"llama\"``,
    ``\"llama2\"`` (GQA). ``model_size`` defaults from ``hf_arch`` for checkpoint adapter
    selection (``gpt_adapter`` vs ``llama_adapter``).
    """

    args = _Namespace()

    if hf_arch not in ("gpt", "llama", "llama2"):
        raise ValueError(f"hf_arch must be gpt, llama, or llama2, got {hf_arch!r}")

    is_llama = hf_arch in ("llama", "llama2")
    if model_size is None:
        model_size = "gpt" if hf_arch == "gpt" else hf_arch

    # --- top-level distributed fields ---
    args.rank = rank
    args.world_size = world_size
    args.local_rank = rank
    args.distributed_backend = "nccl"
    args.distributed_timeout_minutes = 10

    # --- args.parallel ---
    args.parallel = _Namespace(
        pp_deg=1,
        global_tp_deg=1,
        global_tp_consec=1,
        global_cp_deg=1,
        global_ep_deg=1,
        global_tp_of_ep_deg=1,
        global_checkpoint=0,
        cp_mode="zigzag",
        sdp=0,
        default_dp_type="ddp",
        pipeline_type="gpipe",
        galvatron_config_path=galvatron_config_path,
        vocab_sdp=0,
        vocab_tp=1,
        vocab_cp=1,
        vocab_sp=0,
        async_grad_reduce=async_grad_reduce,
        mixed_precision=mixed_precision,
        use_ulysses=use_ulysses,
        reduce_in_fp32=True,
        entropy_in_fp32=True,
    )

    # --- args.model ---
    padded_vocab_size = vocab_size
    kv_channels = hidden_size // num_attention_heads

    n_query_groups = num_query_groups if group_query_attention else None

    args.model = _Namespace(
        model_type="gpt",
        model_size=model_size,
        is_moe_model=False,
        hf_model_name_or_path=None,
        model_config_path=None,
        set_model_config_manually=0,
        set_layernum_manually=0,
        set_seqlen_manually=0,
        initialize_on_meta=True,
        shape_order="SBH",
        dropout_prob=0.0,
        print_loss=0,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_query_groups=n_query_groups,
        kv_channels=kv_channels,
        vocab_size=vocab_size,
        padded_vocab_size=padded_vocab_size,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_qkv_bias=False,
        add_bias_linear=not is_llama,
        layernorm_epsilon=norm_epsilon,
        qk_layernorm=False,
        position_embedding_type="rope" if is_llama else "learned_absolute",
        rotary_base=10000,
        rotary_percent=1.0,
        rotary_interleaved=False,
        rotary_seq_len_interpolation_factor=None,
        mrope_section=None,
        make_vocab_size_divisible_by=1,
        normalization="RMSNorm" if is_llama else "LayerNorm",
        norm_epsilon=norm_epsilon,
        multi_latent_attention=False,
        apply_rope_fusion=False,
        bias_activation_fusion=False,
        activation_func_fp8_input_store=False,
        gated_linear_unit=is_llama,
        activation_func=(
            torch.nn.functional.silu if is_llama else torch.nn.functional.gelu
        ),
        untie_embeddings_and_output_weights=False,
        num_moe_experts=None,
        moe_ffn_hidden_size=None,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
        defer_embedding_wgrad_compute=False,
        wgrad_deferral_limit=0,
    )

    # --- args.train ---
    args.train = _Namespace(
        seed=seed,
        iteration=0,
        train_iters=None,
        train_samples=None,
        lr=1e-5,
        min_lr=None,
        weight_decay=0.01,
        start_weight_decay=None,
        end_weight_decay=None,
        weight_decay_incr_style="constant",
        sequence_parallel=sequence_parallel,
        use_flash_attn=use_flash_attn,
        global_batch_size=global_batch_size,
        micro_batch_size=None,
        chunks=chunks,
        seq_length=seq_length,
        clip_grad=1.0,
        flash_decode=True,
        test_mode=False,
    )

    # --- args.profile ---
    args.profile = _Namespace(
        profile=0,
        profile_mode="static",
        profile_unit="all",
        profile_forward=0,
        save_profiled_memory=0,
        exit_after_profiling=1,
    )

    # --- args.ckpt ---
    args.ckpt = _Namespace(
        load=checkpoint_load,
        load_iteration=0,
        distributed_checkpoint=False,
        save=None,
        save_interval=None,
    )

    # --- args.data ---
    args.data = _Namespace(
        data_path=None,
        split=None,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=None,
        shared_storage=True,
        num_dataset_builder_threads=1,
    )

    # --- args.logging ---
    args.logging = _Namespace(
        tensorboard_dir=None,
        wandb_project="",
        wandb_exp_name="",
        wandb_save_dir="",
    )

    # --- backward-compat flat aliases (checkpoint adapters) ---
    args.padded_vocab_size = padded_vocab_size
    args.hidden_size = hidden_size
    args.num_attention_heads = num_attention_heads
    args.seq_length = seq_length
    args.kv_channels = kv_channels
    args.group_query_attention = group_query_attention
    args.num_query_groups = (
        num_query_groups if group_query_attention else num_attention_heads
    )

    return args
