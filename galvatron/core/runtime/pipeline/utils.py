from typing import List, Optional, Union

import torch


def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]

def chunk_cu_seqlens(cu_seqlens, chunks):
    assert cu_seqlens is not None
    batch_size = cu_seqlens.shape[0] - 1
    
    assert batch_size >= chunks, f'batch_size ({batch_size}) must be greater than or equal to chunks ({chunks}). Consider adjusting chunks before chunking.'

    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    mbsz_per_chunk = (batch_size + chunks - 1) // chunks
    mbsz_last_chunk = batch_size - mbsz_per_chunk * (chunks - 1)

    cu_seqlens_list = []
    start_idx = 0
    for chunk_idx in range(chunks):
        mbsz = mbsz_per_chunk if chunk_idx != chunks - 1 else mbsz_last_chunk
        end_idx = start_idx + mbsz
        curr_cu_seqlens = torch.zeros(mbsz + 1, dtype=cu_seqlens.dtype)
        curr_cu_seqlens[0] = 0
        for i in range(1, mbsz + 1):
            curr_cu_seqlens[i] = curr_cu_seqlens[i - 1] + seq_lengths[start_idx + i - 1]
        cu_seqlens_list.append(curr_cu_seqlens)
        start_idx = end_idx

    return cu_seqlens_list

def chunk_packed_tensor(tensor:torch.Tensor, chunks:int, cu_seqlens:torch.Tensor):
    assert tensor.dim() == 2, f'tensor must be a 2D tensor, but got shape {tensor.shape}'
    assert cu_seqlens is not None and cu_seqlens.dim() == 1, f'cu_seqlens must be a 1D tensor, but got shape {cu_seqlens.shape}'

    batch_size = cu_seqlens.shape[0] - 1
    assert batch_size >= chunks, f'batch_size ({batch_size}) must be greater than or equal to chunks ({chunks}). Consider adjusting chunks before chunking.'

    mbsz_per_chunk = (batch_size + chunks - 1) // chunks
    mbsz_last_chunk = batch_size - mbsz_per_chunk * (chunks - 1)
    assert mbsz_last_chunk > 0, f'mbsz_last_chunk ({mbsz_last_chunk}) must be greater than 0'

    tensor_list = []
    start_idx = 0
    for chunk_idx in range(chunks):
        mbsz = mbsz_per_chunk if chunk_idx != chunks - 1 else mbsz_last_chunk
        end_idx = start_idx + mbsz
        start_token_ids, end_token_ids = cu_seqlens[start_idx], cu_seqlens[end_idx]
        curr_tensor = tensor[:, start_token_ids:end_token_ids].contiguous()
        tensor_list.append(curr_tensor)
        start_idx = end_idx

    return tensor_list

def chunk_input_ids(input_ids:torch.Tensor, chunks:int, cu_seqlens:torch.Tensor=None):
    assert input_ids.dim() == 2, f'input_ids must be a 2D tensor, but got shape {input_ids.shape}'
    assert cu_seqlens is None or cu_seqlens.dim() == 1, f'cu_seqlens must be a 1D tensor, but got shape {cu_seqlens.shape}'

    micro_input_ids:List[torch.Tensor] = []
    if cu_seqlens is None:
        micro_input_ids = list(input_ids.chunk(chunks))
    else:
        micro_input_ids = chunk_packed_tensor(input_ids, chunks, cu_seqlens)
    
    return micro_input_ids

def chunk_kwargs(kwargs, chunks:int):
    """
        kwargs: {
            "cu_seqlens": None | torch.Tensor,
            "attention_mask": attention_mask,
            "labels": labels,
            "rotary_embedding": rotary_embedding,
        }
    """
    kwargs_list = [{} for _ in range(chunks)]
    pack_flag = True if kwargs['cu_seqlens'] is not None else False

    for key, value in kwargs.items():
        if key == 'labels':
            if pack_flag:
                label_list = chunk_packed_tensor(value, chunks, kwargs['cu_seqlens'])
            else:
                label_list = list(value.chunk(chunks))
            for i, label in enumerate(label_list):
                kwargs_list[i][key] = label
        elif key == 'attention_mask' or key == 'rotary_embedding':
            for i in range(chunks):
                kwargs_list[i][key] = value
        elif key == 'cu_seqlens':
            if value is None:
                cu_seqlens_list = [None] * chunks
            else:
                cu_seqlens_list = chunk_cu_seqlens(value, chunks)
            for i, cu_seqlens in enumerate(cu_seqlens_list):
                kwargs_list[i][key] = cu_seqlens

    return kwargs_list


# ============original version=================
def chunk_batch(inputs, chunks):
    if inputs is None:
        return inputs

    batches = [[] for _ in range(chunks)]
    # Actual number of chunks produced
    num_chunks = -1
    for input in inputs:
        if torch.is_tensor(input):
            # Chunk only tensors.
            tensors = input.chunk(chunks)

            # Validate number of chunks equal across all inputs.
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(
                    f"Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}"
                )
            num_chunks = len(tensors)

            for i, tensor in enumerate(tensors):
                batches[i].append(tensor)
        else:
            # Replicate non-tensors or tensors wrapped with 'NoChunk'.
            for i in range(chunks):
                batches[i].append(input)
            num_chunks = chunks

    # Truncate to actual number of chunks
    batches = batches[:num_chunks]

    return batches


def chunk_dict(kwargs, chunks):
    batches = [{} for _ in range(chunks)]
    num_chunks = -1
    for k, v in kwargs.items():
        if torch.is_tensor(v) and not (k.endswith("_mask") and v.shape[0] == 1) and not k.startswith("rotary") and not k.startswith("cu_seqlens"):
            tensors = v.chunk(chunks)
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(
                    f"Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}"
                )
            num_chunks = len(tensors)
            for i, tensor in enumerate(tensors):
                batches[i][k] = tensor
        else:
            for i in range(chunks):
                batches[i][k] = v

    if num_chunks >= 0:
        batches = batches[:num_chunks]
    return batches