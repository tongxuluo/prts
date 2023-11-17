from typing import Optional
import torch
import torch.nn as nn


def expand_tensor(
    tensor: torch.Tensor,  # the tensor needing expansion (a, b)
    trg_shape: torch.Size,  
    extra_src_indices: Optional[torch.Tensor],
    div=True,
    device='cuda:0' 
):
    if extra_src_indices is None:
        return tensor
    assert len(tensor.shape) == len(trg_shape)
    assert len(extra_src_indices.shape) == 1
    
    extra_src_tensor = tensor
    non_align_dim = [dim for dim in range(len(tensor.shape)) if tensor.size(dim) != trg_shape[dim]]

    assert len(non_align_dim) <= 1, 'only support one dimension expansion'

    dim = non_align_dim[0]

    # convert device
    origin_device = tensor.device
    tensor = tensor.to(device)
    extra_src_tensor = extra_src_tensor.to(device)
    extra_src_indices = extra_src_indices.to(device)

    # rand初始化一个new_tensor
    new_tensor = torch.randn(trg_shape, dtype=tensor.dtype, device=device)
    # 赋值原来的部分
    new_tensor.narrow(dim, 0, tensor.size(dim)).copy_(tensor)
    padding_tensor = extra_src_tensor.index_select(dim, extra_src_indices)

    if div:
        # count the repeated times of each element in extra_src_indices
        repeat_count = (extra_src_indices.unsqueeze(-1) == extra_src_indices).sum(dim=-1)
        for _ in range(0, dim):
            repeat_count = repeat_count.unsqueeze(0)
        for _ in range(dim+1, len(padding_tensor.shape)):
            repeat_count = repeat_count.unsqueeze(-1)
        
        if not torch.all(tensor.index_select(dim, extra_src_indices)==padding_tensor):
            print('Warning: div=True and the source of paddings is not tensor, the result may be unexpected')
            # print(tensor.index_select(dim, extra_src_indices)==padding_tensor)
            padding_tensor = padding_tensor / repeat_count
        else:
            repeat_count = repeat_count + 1 # +的这个1是复制的那一份
            padding_tensor = padding_tensor / repeat_count
            # 原来的参数位置也需要除以repeat_count
            new_tensor.index_copy_(dim, extra_src_indices, padding_tensor)
    # print(new_tensor.shape)
    # print(padding_tensor.shape)
    new_tensor.narrow(dim,tensor.size(dim), trg_shape[dim]-tensor.size(dim)).copy_(padding_tensor)
    new_tensor = new_tensor.to(origin_device)
    tensor = tensor.to(origin_device)
    extra_src_tensor = extra_src_tensor.to(origin_device)
    extra_src_indices = extra_src_indices.to(origin_device)
    return new_tensor


def copy_init(meta_module, trg_module, vocab_size):
    if meta_module.weight.shape == trg_module.weight.shape:
        trg_module.weight.data = meta_module.weight.data.clone()
        if hasattr(meta_module, 'bias') and hasattr(trg_module, 'bias') and meta_module.bias is not None:
            trg_module.bias.data = meta_module.bias.data.clone()
    elif isinstance(meta_module, nn.Embedding):
        extra_src_indices = torch.randint(
            0, trg_module.weight.shape[1] - meta_module.weight.shape[1],
            size=(trg_module.weight.shape[1] - meta_module.weight.shape[1],)
            )
        trg_shape = torch.Size((trg_module.weight.shape[0], trg_module.weight.shape[1]))
        weight = expand_tensor(
            tensor=meta_module.weight.data.clone(),
            trg_shape=trg_shape,
            extra_src_indices=extra_src_indices,
            div=False,
            device=meta_module.weight.device
            )
        trg_module.weight.data = weight
    elif isinstance(meta_module, nn.Linear) and meta_module.weight.shape[0] == vocab_size:
        extra_src_indices = torch.randint(
            0, trg_module.weight.shape[1] - meta_module.weight.shape[1],
            size=(trg_module.weight.shape[1] - meta_module.weight.shape[1],)
            )
        trg_shape = torch.Size((trg_module.weight.shape[0], trg_module.weight.shape[1]))
        weight = expand_tensor(
            tensor=meta_module.weight.data.clone(),
            trg_shape=trg_shape,
            extra_src_indices=extra_src_indices,
            div=True,
            device=meta_module.weight.device
            )
        trg_module.weight.data = weight
        if meta_module.bias is not None:
            trg_module.bias.data = meta_module.bias.data.clone()


def get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def split_layers(modulelist, split_list):
    split_module_lists = []
    start = 0
    for num_modules in split_list:
        split_module_list = nn.ModuleList(modulelist[start:start+num_modules])
        split_module_lists.append(split_module_list)
        start += num_modules
    return split_module_lists


def find_moduleList(model):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        parent, layer, layer_name = get_submodules(model, key)
        if isinstance(layer, nn.ModuleList):
            return parent, layer, layer_name


def make_only_trg_as_trainable(model: nn.Module):
    for n, p in model.named_parameters():
        if "trg_layers" not in n and "global_W" not in n:
            p.requires_grad = False


def make_only_before_n_layer_trg_as_trainable(
    model: nn.Module,
    activate_before_n_layer: int,
):
    now_layer_id = -1
    for n, p in model.named_parameters():
        if "trg_layers" in n:
            if "wte" in n:
                now_layer_id = max(now_layer_id, -1)
            elif "ln_f" in n and "lm_head" in n:
                now_layer_id = max(now_layer_id, 1e3)
            else:
                now_layer_id = max(now_layer_id, int(n.split(".h.")[-1].split(".")[0]))
    deactivate_layer_id = now_layer_id - activate_before_n_layer
    for n, p in model.named_parameters():
        if "trg_layers" in n:
            if "wte" in n and -1 < deactivate_layer_id:
                p.requires_grad = False
            elif "ln_f" in n and "lm_head" in n:
                continue
            else:
                layer_id = int(n.split(".h.")[-1].split(".")[0])
                if layer_id < deactivate_layer_id:
                    p.requires_grad = False


def switch_key(key: str, block_layers: int, block_idx: Optional[int] = None) -> str:
    key = key.replace("meta_model.", "")
    keys = key.split(".")
    # 查找 "trg_layers" 在列表中的索引
    index_trg_layers = keys.index("trg_layers")

    if index_trg_layers < len(keys) - 1:
        if block_idx is None:
            block_idx = int(keys.pop(index_trg_layers - 1))
        else:
            keys.pop(index_trg_layers - 1)
        keys.pop(index_trg_layers - 1)
        keys[index_trg_layers - 1] = str(int(keys[index_trg_layers - 1]) + block_idx * block_layers)
    print(keys)
    return ".".join(keys)