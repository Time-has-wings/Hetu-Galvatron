from galvatron.core.cost_model.arguments import galvatron_cost_model_args

def galvatron_search_args(parser):
    # Integrate the arguments of cost model into the arguments of search engine
    parser = galvatron_cost_model_args(parser)

    group = parser.add_argument_group(title="Galvatron Searching Arguments")
    
    # system info
    group.add_argument(
        "--memory_constraint", type=int, default=24, help="Memory constraint of Galvatron",
    )
    group.add_argument(
        "--output_config_path", type=str, default='./configs', help="Path to output config."
    )
    group.add_argument(
        "--fine_grained_mode", type=int, default=1, help="Enable fine-grained search."
    )
    group.add_argument(
        "--parallel_search", action="store_true", help="Enable parallel search for faster execution"
    )
    group.add_argument(
        "--worker", type=int, default=0, help="Number of worker threads for parallel search. Default is 2x CPU cores"
    )
    group.add_argument(
        "--log_dir", type=str, default="search_engine_logs", help="Log directory for search engine."
    )

    # bsz related args
    group.add_argument(
        "--min_bsz", type=int, default=8, help="Min batch size for searching.",
    )
    group.add_argument(
        "--max_bsz", type=int, default=10240, help="Max batch size for searching.",
    )
    group.add_argument(
        "--recommend_min_bsz", type=int, default=0, help="If 1, start searching from a recommended bsz to accelerate optimization.",
    )
    group.add_argument(
        "--settle_bsz", type=int, default=-1, help="If > 1, only search bsz=settle_bsz."
    )
    group.add_argument(
        "--settle_chunk", type=int, default=-1, help="If > 1, only search chunk=settle_chunk."
    )
    group.add_argument(
        "--bsz_scale", type=int, default=8, help="Bsz scale for searching.",
    )

    # strategy related args
    group.add_argument(
        "--disable_pp", type=int, default=0, help="Whether to disable pp."
    )
    group.add_argument(
        "--disable_tp", type=int, default=0, help="Whether to disable tp."
    )
    group.add_argument(
        "--disable_sp", type=int, default=0, help="Whether to disable sp."
    )
    group.add_argument(
        "--disable_cp", type=int, default=0, help="Whether to disable cp."
    )
    group.add_argument(
        "--disable_dp", type=int, default=0, help="Whether to disable dp."
    )
    group.add_argument(
        "--disable_ckpt", type=int, default=0, help="Whether to disable checkpoint"
    )
    group.add_argument(
        "--disable_fsdp", type=int, default=0, help="Whether to disable fsdp."
    )
    group.add_argument(
        "--disable_embedding_lmhead_tp", type=int, default=0, help="Whether to disable embedding and lmhead tp."
    )
    group.add_argument(
        "--disable_embedding_lmhead_sp", type=int, default=0, help="Whether to disable embedding and lmhead sp."
    )
    group.add_argument(
        "--disable_tp_consec", type=int, default=0, help="Whether to disable tp_consec."
    )
    group.add_argument(
        "--max_pp_deg", type=int, default=-1, help="Maximum pipeline parallel degree to search."
    )
    group.add_argument(
        "--max_tp_deg", type=int, default=-1, help="Maximum tensor parallel degree to search."
    )
    group.add_argument(
        "--max_sp_deg", type=int, default=-1, help="Maximum tensor parallel degree to search."
    )
    group.add_argument(
        "--max_cp_deg", type=int, default=-1, help="Maximum tensor parallel degree to search."
    )

    # train config args
    group.add_argument(
        "--default_dp_type", type=str, default="ddp", help="Default data parallel type", choices=["ddp","zero2"],
    )
    group.add_argument(
        "--no_global_memory_buffer", action="store_false", help='Disable the estimation of global memory for all gather buffer when using Megatron-SP.', dest='global_memory_buffer'
    )
    
    return parser
