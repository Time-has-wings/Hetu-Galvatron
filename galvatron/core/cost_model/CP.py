def cp_metrics(seq_len: int, 
               head_dim: int,
               num_attention_heads:int,
               num_query_groups:int,
               bsz: int, 
               cp_size: int,
                 a: float, 
                 b: float, 
                 c: float,
                mixed_precision = True):
    MB = 1024.0 * 1024.0
    pp_bandwidth = 78.0232 * 1.024
    bytes_kv = 2 if mixed_precision else 4
    bytes_fp32 = 4
   
    x = seq_len # x是seq_len

    # 单独求解计算时间
    # compute per-step (与 cost_model 一致)：seq->seq/p，再乘 (bsz/tp)
    # 这个计算时间不太对吧。->不对，这个计算时间是对的。这个计算时间其实只需要考虑每一次小批次的attention的计算
    comp_fwd = (a * x * x + b * x + c) / cp_size # 对seq_len下的计算量进行二次拟合，随后再除以cp_size来说明被cp_size进行了均分
    comp_bwd = comp_fwd * 2 # 反向计算时间是前向计算时间的两倍
   
    # 单独求解计算时间
    Q_elems_per_step = (bsz * seq_len * head_dim * num_attention_heads) / cp_size
    K_elems_per_step = (bsz * seq_len * head_dim * num_query_groups) / cp_size
    V_elems_per_step = (bsz * seq_len * head_dim * num_query_groups) / cp_size
   
    # 前向：以 K/V 大小为主
    comm_fwd_ms = ( K_elems_per_step + V_elems_per_step) * bytes_kv / MB / pp_bandwidth * (cp_size - 1)
    # 反向首段：K/V 相关
    comm_bwd_0_ms = ( K_elems_per_step + V_elems_per_step) * bytes_kv / MB / pp_bandwidth
    # 反向中段/尾段：以 Q/O 梯度大小为主
    comm_bwd_mid_ms = 2*( K_elems_per_step + V_elems_per_step) * bytes_fp32 / MB / pp_bandwidth * (cp_size - 1)
    comm_bwd_tail_ms = ( K_elems_per_step + V_elems_per_step) * bytes_fp32 / MB / pp_bandwidth
    comm_bwd_ms = comm_bwd_0_ms + comm_bwd_mid_ms + comm_bwd_tail_ms

    # 计算与通信进行重叠的降速。说实话，这个东西是一定会有问题的。因为存在太多的计算与通信进行重叠，然后这个顺序又没有很好的构建清晰
    # overlap 函数（等价于 cost_model 的 overlap_comp_comm_time）
    def overlap_comp_comm_time(comp, comm):
        if comp >= comm:
            return comp + 0.1 * comm # 0.1是一个经验值，视为计算与通信重叠的降速
        return comm

    # forward 总时间（每层）
    fwd_per_layer_ms = overlap_comp_comm_time(comp_fwd, comm_fwd_ms)
    
    bwd_per_layer_ms = overlap_comp_comm_time(comp_bwd, comm_bwd_ms)
   
    total_ms = (fwd_per_layer_ms + bwd_per_layer_ms) 
    comm = comm_fwd_ms + comm_bwd_ms
    return  comm, total_ms