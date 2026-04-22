import os

import torch

from galvatron.utils.config_utils import num2str, read_json_config, write_json_config


def print_peak_memory(prefix, device, type="allocated"):
    if type == "allocated":
        print(prefix, "[Allocated]")
        max_mem = torch.cuda.max_memory_allocated(device) / 2**20
        cur_mem = torch.cuda.memory_allocated(device) / 2**20
        print("\tMax memory: %.2f MB\tCurrent memory : %.2f MB" % (max_mem, cur_mem))
    elif type == "reserved":
        print(prefix, "[Reserved]")
        max_mem = torch.cuda.max_memory_reserved(device) / 2**20
        cur_mem = torch.cuda.memory_reserved(device) / 2**20
        print("\tMax memory: %.2f MB\tCurrent memory : %.2f MB" % (max_mem, cur_mem))
    return max_mem, cur_mem


def save_profiled_memory(
    path,
    pp_deg,
    tp_deg,
    world_size,
    layer_num,
    bsz,
    rank,
    model_states,
    activation,
    activation_peak,
    cpt,
    sequence_parallel=False,
    vocab_tp=1,
    seq=None,
):
    config = read_json_config(path) if os.path.exists(path) else {}
    key = "%d_%d_%d" % (pp_deg, tp_deg, world_size // pp_deg // tp_deg)
    if cpt:
        key += "_c"
    if vocab_tp == tp_deg and tp_deg != 1:
        key += "_vtp"
    if sequence_parallel:
        key += "_sp"
    if key not in config.keys():
        config[key] = {}
    layernum_info = num2str(layer_num, "layernum")
    seq_info = num2str(seq, "seq")
    config[key][f"{layernum_info}_bsz{bsz}_{seq_info}_rank{rank}_ms"] = model_states
    config[key][f"{layernum_info}_bsz{bsz}_{seq_info}_rank{rank}_act"] = activation
    config[key][f"{layernum_info}_bsz{bsz}_{seq_info}_rank{rank}_act_peak"] = activation_peak
    write_json_config(config, path)
    print("Already written profiled memory into config file %s!\n" % (path))


def save_profiled_time(path, time, bsz, layer_num, seq):
    config = read_json_config(path) if os.path.exists(path) else {}
    layernum_info = num2str(layer_num, "layernum")
    seq_info = num2str(seq, "seq")
    key = f"{layernum_info}_bsz{bsz}_{seq_info}"
    config[key] = time
    write_json_config(config, path)
    print("Already written profiled time into config file %s!\n" % (path))
