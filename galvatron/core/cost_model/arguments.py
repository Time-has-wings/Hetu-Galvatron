def galvatron_cost_model_args(parser):
    group = parser.add_argument_group(title="Galvatron Cost Model Arguments")
    
    # model args
    group.add_argument(
        "--set_model_config_manually", type=int, default=0, help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten."
    )
    group.add_argument(
        "--set_layernum_manually", type=int, default=0, help="Whether to set layernum config manually (doesn't overwrite other model configs)."
    )
    group.add_argument(
        "--set_seqlen_manually", type=int, default=0, help="Whether to set sequence length config manually (doesn't overwrite other model configs)."
    )
    group.add_argument(
        "--set_experts_manually", type=int, default=0, help="Whether to set experts config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--num_experts", type=int, default=8, help="Total number of experts in each MoE (Mixture of Experts) layer. Defaults to 8."
    )
    group.add_argument(
        "--top_k", type=int, default=2, help="Number of top experts to select for each token in MoE (Mixture of Experts) layers. Defaults to 2."
    )
    group.add_argument(
        "--moe_grouped_gemm", type=int, default=0, help="Whether to use grouped GEMM optimization for MoE computations (1 for enable, 0 for disable)."
    )
    group.add_argument(
        "--is_MoE", type=int, default=0, help="Whether the model uses Mixture of Experts (MoE) architecture. Set to 1 if using MoE, 0 otherwise. Defaults to 0."
    )

    # cluster info
    group.add_argument(
        "--num_nodes", type=int, default=1, help="Number of Nodes.",
    )
    group.add_argument(
        "--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node.",
    )

    # train args
    group.add_argument(
        "--mixed_precision", type=str, default="bf16", help="Mixed precision option.", choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument(
        "--pipeline_type", type=str, default="gpipe", help="Galvatron pipeline type", choices=["gpipe","pipedream_flush"],
    )
    group.add_argument(
        "--sequence_parallel", action="store_true", help="Whether to use sequence parallel",
    )
    group.add_argument(
        "--async_grad_reduce", type=int, default=1, help='Whether to async grad reduce so that gradient will be reduce every micro batch. Ensure Zero3 memory cost when chunk > 1.',
    )

    # profile hardware 
    group.add_argument(
        "--allreduce_bandwidth_config_path", type=str, default=None, help="Path to allreduce bandwidth config."
    )
    group.add_argument(
        "--p2p_bandwidth_config_path", type=str, default=None, help="Path to p2p bandwidth config."
    )
    group.add_argument(
        "--overlap_coe_path", type=str, default=None, help="Path to overlap coefficient config."
    )
    group.add_argument(
        "--sp_time_path", type=str, default=None, help="Path to sequence parallelism time config."
    )

    # profile model (file path, profile mode and profile granularity)
    group.add_argument(
        "--memory_profiling_path", type=str, default=None, help="Path to memory profiling config."
    )
    group.add_argument(
        "--time_profiling_path", type=str, default=None, help="Path to time profiling config."
    )
    group.add_argument(
        "--time_profile_mode", type=str, nargs="+", default=["batch"], help="Galvatron profiling mode", choices=["static", "batch", "sequence"]
    )
    group.add_argument(
        "--memory_profile_mode", type=str, nargs="+", default=["static"], help="Galvatron profiling mode", choices=["static", "sequence"]
    )
    group.add_argument(
        "--profile_granularity", type=str, default="together", help="Granularity of profiling.", choices=["together", "split"]
    )   

    # version option args
    group.add_argument(
        "--estimate_tp_time_type", type=str, default="fixed", help="Estimate tensor parallel time by fixed value or fitted curve.", choices=["fixed", "fit"]
    )
    group.add_argument(
        "--zero_with_slight_noise", type=int, default=1, help="Whether to add slight noise to zero memory estimation to improve optimization stability."
    )

    # utils args
    group.add_argument(
            "--costmodel_coe", type=float, default=1.0, help="Multiply the outcome of time cost model by this coefficient. Only for fine-tuning time cost model, should be 1.0 in default.",
        )
    return parser
