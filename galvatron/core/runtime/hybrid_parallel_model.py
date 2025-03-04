import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed import fsdp

from .comm_groups import gen_comm_groups
from .hybrid_parallel_config import (
    check_hp_config,
    get_chunks,
    get_enc_groups,
    hp_config_whole_model,
    layer_shapes_dtypes_whole_model,
    mixed_precision_dtype,
)
from .initialize import init_empty_weights
from .parallel import wrap_modules_relocation
from .pipeline.grad_reduce import _finalize_params_bf16, _register_post_backward_hook_bf16
from .utils import get_layernorm_offset

version_str = torch.__version__
version_major, version_minor, _ = version_str.split(".")
version_major, version_minor = int(version_major), int(version_minor)
if version_major > 1:
    if version_minor > 0:
        from torch.distributed.fsdp._runtime_utils import _register_post_backward_hook

    else:
        from torch.distributed.fsdp._runtime_utils import _register_post_backward_hooks
else:
    assert False, f"PyTorch version must be greater than 2.0, but found {torch.__version__}"


# [note] 该类仅仅在construct_hybrid_parallel_model_api被调用一次
class GalvatronModel(nn.Module):
    def __init__(self, hp_model):
        super().__init__()
        from galvatron.core import get_args

        self.args = get_args()
        self.model = hp_model # 设置混合并行模型
        self.iter = 0

    def forward_backward(self, batch, iter=None, profiler=None, loss_func=None, **kwargs):
        """_summary_

        执行前向和反向传播，支持流水线并行和非流水线模式。

        参数:
            batch: 输入批次数据
            iter (int, 可选): 当前迭代次数，默认为None
            profiler: 性能分析器，默认为None
            loss_func (callable, 可选): 损失函数，默认为None
            **kwargs: 其他关键字参数

        返回:
            loss: 计算得到的损失值（移至CPU）
        """
        
        args, model = self.args, self.model
        self.iter = iter if iter is not None else self.iter
        
        # 如果提供了损失函数，确保batch格式正确
        if loss_func is not None:
            if len(batch) == 1 and isinstance(batch[0], Tensor):
                batch = [batch, [self.fake_tensor(batch[0])]]
            assert (
                isinstance(batch, (tuple, list))
                and isinstance(batch[0], (tuple, list))
                and isinstance(batch[1], (tuple, list))
            )
        else: # 如果未提供损失函数，使用伪损失函数
            loss_func = self.fake_loss_func
            assert isinstance(batch, (tuple, list))
            batch = [batch, [self.fake_tensor(batch[0])]]
            
        # 根据流水线并行度选择执行模式
        if args.pp_deg > 1:
            if args.pipeline_type == "gpipe":
                loss = model.gpipe_forward(batch, loss_func, **kwargs)
                if profiler is not None:
                    profiler.profile_memory(self.iter, "After Forward")
                model.gpipe_backward()
            elif args.pipeline_type == "pipedream_flush":
                loss = model.pipedream_flush_forward_backward(batch, loss_func, **kwargs)
        else:
            loss = model.no_pipeline_forward_backward(
                batch, loss_func, forward_only=args.profile_forward, profiler=profiler, iter=self.iter, **kwargs
            )
        self.iter += 1
        return self.loss_to_cpu(loss)

    def fake_tensor(self, x):
        return torch.zeros([x.shape[0], 1], dtype=x.dtype, device=x.device)

    def fake_loss_func(self, labels, outputs):
        if torch.numel(outputs[0]) > 1:
            loss = outputs[0].mean()
            return loss, loss
        return outputs[0], outputs[0]

    def loss_to_cpu(self, loss):
        if isinstance(loss, (list, tuple)):  # Average loss of each microbatch
            if len(loss) == 0:
                return None
            loss = np.mean([l.item() for l in loss])
        else:
            loss = loss.item()
        return loss


class GalvatronModelWrapper:
    def __init__(self, args, wrap_block_names=[]):
        """
        初始化Galvatron模型包装器。

        参数:
            args: 命令行参数对象
            wrap_block_names (list, 可选): 需要包装的模块名称列表，默认为空
        """
        self.args = args
        self.wrap_block_names = wrap_block_names

    # Wrap Galvatron Hybrid Parallel Model, need to be called after Galvatron is initialized
    def wrap_model_hybrid_parallel(
        self,
        model,
        model_config,
        hybrid_parallel_configs,
        model_info,
        construct_sequential_model,
        construct_tensor_parallel_model,
    ):
        """
        包装混合并行模型，在Galvatron初始化后调用。

        参数:
            model: 原始模型
            model_config: 模型配置
            hybrid_parallel_configs (dict): 混合并行配置
            model_info: 模型信息对象
            construct_sequential_model (callable): 构建顺序模型的函数
            construct_tensor_parallel_model (callable): 构建张量并行模型的函数

        返回:
            GalvatronModel: 包装后的混合并行模型
        """
        return construct_hybrid_parallel_model_api(
            model,
            model_config,
            self.args,
            hybrid_parallel_configs,
            model_info,
            construct_sequential_model,
            construct_tensor_parallel_model,
            self.wrap_block_names,
        )

    # [note] 该函数没有被调用过，貌似用于快速调试
    # Wrap Data Parallel Model, can be called on any PyTorch Model even when Galvatron is not initilized
    def wrap_model_data_parallel(
        self,
        model,
        device,
        dp_type="ddp",
        mixed_precision="bf16",
        comm_group=None,
        initialize_on_meta=False,
        backward_prefetch=True,
    ):
        from galvatron.core.parallel import wrap_model_data_parallel

        mixed_precision = mixed_precision_dtype(mixed_precision)
        return wrap_model_data_parallel(
            model,
            device,
            self.wrap_block_names,
            dp_type,
            mixed_precision,
            comm_group,
            initialize_on_meta,
            backward_prefetch,
        )

    # [note] 该函数没有被调用过，貌似用于快速调试
    # Wrap Activation Checkpoint Model, can be called on any PyTorch Model even when Galvatron is not initilized
    def wrap_model_checkpoint(self, model):
        from galvatron.core.parallel import wrap_model_checkpoint

        return wrap_model_checkpoint(model, self.wrap_block_names)

# 定义混合并行模型构建函数
def construct_hybrid_parallel_model_api(
    model,
    model_config,
    training_args,
    hybrid_parallel_configs,
    model_info,
    construct_sequential_model,
    construct_tensor_parallel_model,
    wrap_block_name=None,
    wrap_checkpoint_block_name=None,
    wrap_other_block_name=None,
    tied_wte_attr_names=None,
    layernorm_name=[],
    all_block_name=None,
    load_module_func=None,
    meta_init_buffer=True,
):
    """
    构建混合并行模型，整合张量并行、流水线并行和数据并行。

    参数:
        model: 原始模型
        model_config: 模型配置
        training_args: 训练参数
        hybrid_parallel_configs (dict): 混合并行配置
        model_info: 模型信息对象
        construct_sequential_model (callable): 构建顺序模型的函数
        construct_tensor_parallel_model (callable): 构建张量并行模型的函数
        wrap_block_name (list, 可选): 并行包装块名称
        wrap_checkpoint_block_name (list, 可选): 检查点包装块名称
        wrap_other_block_name (list, 可选): 其他包装块名称
        tied_wte_attr_names (list, 可选): 绑定的词嵌入属性名称
        layernorm_name (list, 可选): LayerNorm模块名称
        all_block_name (list, 可选): 所有块名称
        load_module_func (callable, 可选): 模块加载函数
        meta_init_buffer (bool, 可选): 是否初始化缓冲区，默认为True

    返回:
        GalvatronModel: 构建完成的混合并行模型
    """
    
    if wrap_checkpoint_block_name == None:
        wrap_checkpoint_block_name = wrap_block_name
        
    # 换个名字
    config, args, hp_configs = model_config, training_args, hybrid_parallel_configs

    # 如果使用bf16混合精度，替换FSDP的默认函数以支持bf16
    # _register_post_backward_hook_bf16 和 _finalize_params_bf16 是两个自定义的函数
    if args.mixed_precision == "bf16":
        assert version_major > 1 and version_minor > 0, "Mixed precision training is only supported for torch > 2.0.1"
        fsdp._runtime_utils._register_post_backward_hook = _register_post_backward_hook_bf16
        fsdp._runtime_utils._finalize_params = _finalize_params_bf16
    
    # 获取模型信息：模块类型、层数、层形状和数据类型 # [note]去了解一下这些信息是如何获取的
    # Get model-specific model info: module_types, layernum_list, layer_shapes_list, layer_dtypes_list
    model_info = model_info(config, args)
    module_types = model_info.module_types()
    layernum_list = model_info.layernums()
    layer_shapes_list = model_info.shapes()
    layer_dtypes_list = model_info.dtypes()

    # 检查混合并行配置的有效性（仅检查编码器部分）
    # Check the validity of hp_configs (encoders only)
    check_hp_config(hp_configs, layernum_list)

    # 计算整个模型的形状和数据类型（包括嵌入层等）
    # Calculate shapes and dtypes for whole model (including embed/cls/... layers)
    shapes_whole, dtypes_whole = layer_shapes_dtypes_whole_model(
        module_types, layernum_list, layer_shapes_list, layer_dtypes_list
    )

    # 生成整个模型的混合并行配置   # [note] # [confused] 貌似是因为对每一层都进行配置而带来的
    # Get hp_configs_whole for the whole model (including embed/cls/... layers)
    hp_configs_whole = hp_config_whole_model(
        module_types, hp_configs, embed_sdp=args.embed_sdp, embed_ckpt=0, vocab_tp=args.vocab_tp, vocab_sp=args.vocab_sp
    )

    # if args.use_ulysses:
    #     hp_configs_whole['sp_sizes_whole'] = hp_configs_whole['tp_sizes_whole']
    #     hp_configs_whole['tp_sizes_whole'] = [1] * len(hp_configs_whole['tp_sizes_whole'])
    # else:
    #     hp_configs_whole['sp_sizes_whole'] = [1] * len(hp_configs_whole['tp_sizes_whole'])

    # [Step 0] Generate communication groups
    (
        pp_group,
        tp_groups_whole,
        sp_groups_whole,
        dp_groups_whole,
        seq_data_groups_whole,
        allgather_groups_whole,
        split_groups_whole,
        fused_allgather_groups_whole,
        fused_split_groups_whole,
        embedding_group,
    ) = gen_comm_groups(
        hp_configs_whole["tp_sizes_whole"],
        hp_configs_whole["sp_sizes_whole"],
        hp_configs_whole["pp_deg"],
        hp_configs_whole["tp_consec_whole"],
        show_rank=0,
    )

    # [Step 1] 构建张量并行模型
    # construct_tensor_parallel_model函数是本函数的一个参数
    # [Step 1] Construct Tensor Parallel Model based on tp_groups using model-specific TP function
    if args.initialize_on_meta and args.shape_order == "SBH":
        with init_empty_weights(meta_init_buffer):
            model = construct_tensor_parallel_model(model, config, tp_groups_whole, sp_groups_whole)
    elif args.shape_order == "SBH":
        model = construct_tensor_parallel_model(model, config, tp_groups_whole, sp_groups_whole)
    else:
        assert not args.use_ulysses, "FA model does not support ulysses!"
        model = construct_tensor_parallel_model(model, config, tp_groups_whole)
    
    # 构建序列并行
    # construct_sequential_model函数是本函数的一个参数
    # [Step 2] Construct Sequantial model using model-specific sequential function
    if args.initialize_on_meta and args.shape_order == "SBH":
        with init_empty_weights(meta_init_buffer):
            model = construct_sequential_model(model, config)
    else:
        model = construct_sequential_model(model, config)

    # [Step 3] 包装重定位模块 #[note]学习一下relocation是什么 wrap_modules_relocation函数是parallel.py中的函数
    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(
        model, allgather_groups_whole, split_groups_whole, fused_allgather_groups_whole, fused_split_groups_whole
    )
    # 获取LayerNorm的偏移和大小
    ln_offset, ln_size = get_layernorm_offset(model, layernorm_name)
    assert len(ln_offset) == len(dp_groups_whole)
    
    # [important] 以下就均为和PipelineParallel类相关的了
    # [Step 4] Construct Pipeline Module and place the layers on corresponding devices
    from galvatron.core.runtime.pipeline import PipelineParallel

    hp_model = PipelineParallel(
        model=model,
        model_ranks=hp_configs_whole["pp_ranks_whole"],
        layer_output_tensor_shapes=shapes_whole,
        layer_output_tensor_dtypes=dtypes_whole,
        layer_dp_sizes=hp_configs_whole["dp_sizes_whole"],
        layer_tp_sizes=hp_configs_whole["tp_sizes_whole"],
        layer_sp_sizes=hp_configs_whole["sp_sizes_whole"],
        chunks=get_chunks(args),
        process_group=pp_group.ranks,
        embedding_group=embedding_group,
        nproc_per_node=8,
        info=False,
        tied_wte_attr_names=tied_wte_attr_names,
    )

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    hp_model.wrap_pipeline_modules_data_parallel(
        hp_configs_whole["dp_types_whole"],
        seq_data_groups_whole,
        module_types=module_types,
        mixed_precision=mixed_precision_dtype(args.mixed_precision),
        wrap_block_name=wrap_block_name,
        wrap_other_block_name=wrap_other_block_name,
        tp_groups=tp_groups_whole,
        all_block_name=all_block_name,
        load_module_func=load_module_func,
    )

    hp_model.gen_sp_layernorm_info(
        layer_module_types=module_types,
        layer_tp_groups=tp_groups_whole,
        ln_offset=ln_offset,
        ln_size=ln_size,
        all_block_name=all_block_name,
    )

    # [Step 6] Wrap checkpoint based on checkpoint_flags
    hp_model.wrap_pipeline_modules_checkpoint(
        hp_configs_whole["checkpoint_flags_whole"], wrap_block_name=wrap_checkpoint_block_name
    )

    model = GalvatronModel(hp_model)

    model.dp_groups_whole = dp_groups_whole
    model.tp_groups_whole = tp_groups_whole
    model.sp_groups_whole = sp_groups_whole
    model.sdp_groups_whole = seq_data_groups_whole
    model.hybrid_parallel_configs = hybrid_parallel_configs

    # 返回model
    return model
