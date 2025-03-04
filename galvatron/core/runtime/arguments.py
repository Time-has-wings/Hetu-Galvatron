def galvatron_training_args(parser, use_megatron=True):
    group = parser.add_argument_group(title="Galvatron Training Arguments")

    group.add_argument( # 是否手动设置模型配置（覆盖基于 model_size 的配置）
        "--set_model_config_manually",
        type=int,
        default=0,
        help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten.",
    )
    group.add_argument( # 是否手动设置层数（不覆盖其他模型配置）
        "--set_layernum_manually",
        type=int,
        default=0,
        help="Whether to set layernum config manually (doesn't overwrite other model configs).",
    )
    group.add_argument( # 是否手动设置序列长度（不覆盖其他模型配置）
        "--set_seqlen_manually",
        type=int,
        default=0,
        help="Whether to set sequence length config manually (doesn't overwrite other model configs).",
    )
    
    group.add_argument( # 是否在元设备上初始化参数 # [note] 学习一下元设备初始化参数是干啥的
        "--initialize_on_meta",
        type=int,
        default=0,
        help="Whether to initialize parameters on meta device.",
        choices=[0, 1],
    )
    group.add_argument("--global_train_batch_size", type=int, default=32, help="Global training batch size")
    group.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout rate.")
    group.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    group.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam")
    group.add_argument("--check_loss", type=int, default=0, help="Whether to check model correctness.") # 是否检查模型正确性
    group.add_argument("--profile", type=int, default=0, help="Whether to profile model GPU memory.") # 是否分析模型 GPU 内存
    group.add_argument("--save_profiled_memory", type=int, default=0, help="Whether to save profiled memory.")
    group.add_argument( # 分析内存类型（已分配或保留）# [note] 学习记录下这两种类型
        "--profile_type",
        type=str,
        default="allocated",
        help="Profile allocated memory or reserved memory.",
        choices=["allocated", "reserved"],
    )
    group.add_argument( # Galvatron 分析模式（静态、批处理、序列）# [note]学习一下profile的构建是如何设置的，参数的含义到底是什么
        "--profile_mode",
        type=str,
        default="static",
        help="Galvatron profiling mode",
        choices=["static", "batch", "sequence"],
    )
    group.add_argument("--load_params", type=int, default=0, help="Whether to load saved init params.")
    group.add_argument(
        "--pp_deg",
        type=int,
        default=2,
        help="Pipeline parallel degree.",
        choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    group.add_argument(
        "--global_tp_deg",
        type=int,
        default=-1,
        help="Global tensor parallel degree.",
        choices=[-1, 1, 2, 4, 8, 16, 32],
    )
    group.add_argument( # 流水线分块数 # [note] 确认chunk的含义
        "--chunks",
        type=int,
        default=-1,
        help="Pipeline chunk num.",
    )
    group.add_argument( # 全局张量并行组连续标志 # [note] 确认连续标志的含义
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    group.add_argument(
        "--sdp",
        type=int,
        default=0,
        help="Apply SDP (zero-3)",
        choices=[0, 1],
    )
    group.add_argument( # Galvatron 策略配置文件路径
        "--galvatron_config_path",
        type=str,
        default=None,
        help="Galvatron strategy config path. If not None, galvatron will run according to json config file.",
    )
    group.add_argument("--global_checkpoint", type=int, default=0, help="Global checkpoint flag.")
    group.add_argument( # 混合精度选项 [important]
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Mixed precision option.",
        choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument( # Galvatron 流水线类型:支持gpipe和pipedream_flush
        "--pipeline_type",
        type=str,
        default="gpipe",
        help="Galvatron pipeline type",
        choices=["gpipe", "pipedream_flush"],
    )
    group.add_argument(
        "--default_dp_type",
        type=str,
        default="ddp",
        help="Default data parallel type",
        choices=["ddp", "zero2", "zero3"],
    )
    group.add_argument( # 是否对嵌入层和分类层应用 SDP（Zero-3）# [note] 没有接触过这个
        "--embed_sdp",
        type=int,
        default=0,
        help="Apply SDP (zero-3) for Embeddings and cls",
        choices=[0, 1],
    )
    group.add_argument( # 是否分析前向传播计算
        "--profile_forward",
        type=int,
        default=0,
        help="Profile forward computation",
        choices=[0, 1],
    )
    group.add_argument( # 是否在 Ampere 设备上允许 TF32 # [note] what this
        "--allow_tf32",
        type=int,
        default=1,
        help="Whether to allow tf32 on Ampere devices",
        choices=[0, 1],
    )
    group.add_argument( # 分析完成后是否退出
        "--exit_after_profiling",
        type=int,
        default=1,
        help="Whether to exit after profiling time and memory.",
        choices=[0, 1],
    )
    group.add_argument( # 模型形状顺序 # 可选值：序列-批次-隐藏，批次-序列-隐藏 #[note]没有看懂
        "--shape_order",
        type=str,
        default="SBH",
        help="Model shape order.",
        choices=["SBH", "BSH"],
    )
    group.add_argument( # 词汇表张量并行度 # [note] what this
        "--vocab_tp",
        type=int,
        default=1,
        help="Tensor parallel degree of vocab.",
        choices=[1, 2, 4, 8, 16],
    )
    group.add_argument(
        "--use-ulysses",
        action="store_true",
        help="Whether to use DeepSpeed Ulysses or Megatron-TP",
    )
    group.add_argument( # 是否禁用异步梯度归约 #[note]增添学习
        "--no_async_grad_reduce",
        action="store_false",
        help="Disable async grad reduce so that gradient will be reduce every micro batch. "
        "Ensure Zero3 memory cost when chunk > 1.",
        dest="async_grad_reduce",
    )
    group.add_argument( # 是否使用 FP32 进行梯度归约
        "--reduce_in_fp32",
        action="store_true",
        help="Use fp32 for gradient reduction.",
    )
    group.add_argument( # 是否使用 FP32 计算熵
        "--entropy_in_fp32",
        action="store_true",
        help="Use fp32 for entropy calculation.",
    )
    group.add_argument( # 是否使用分布式检查点
        "--distributed_checkpoint",
        action="store_true",
        default=False,
        help="Whether to use distributed checkpoint.",
    )
    group.add_argument( # 加载的迭代次数
        "--load_iteration",
        type=int,
        default=0,
        help="Load iteration number.",
    )
    if not use_megatron:
        group.add_argument("--lr", type=float, default=1e-4, help="Learning rate of adam")
        group.add_argument("--gpu_id", type=int, default=0, help="Id of GPU to run.")
        group.add_argument("--local-rank", type=int, default=0, help="Local rank.")
    else:
        group.add_argument("--local-rank", type=int, default=-1, help="Local rank.")
        group.add_argument("--no-shared-storage", action="store_false", dest="shared_storage", help="Cluster is not shared storage.")
    return parser
