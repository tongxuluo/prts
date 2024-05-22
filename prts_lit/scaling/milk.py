from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import copy
from ..utils import (
    MilkConfig,
    MaskSchedule,
    get_submodules,
    split_layers,
    find_moduleList,
    make_only_trg_as_trainable
)


class IncubationLayer(nn.Module):
    def __init__(
        self,
        meta_layers: Union[nn.ModuleList, nn.Linear, nn.Embedding, None],
        trg_layers: Union[nn.ModuleList, nn.Linear, nn.Embedding],
        global_mask_schedule,
    ) -> None:
        super().__init__()
        self.meta_layers = meta_layers
        self.trg_layers = trg_layers
        self.global_mask_schedule = global_mask_schedule
    
    def decoder_layers(
        self,
        layers,
        x: torch.Tensor,
        *args,
        **kwargs
        ) -> Tuple[torch.Tensor, any]:
        for block in layers:
            x, *_ = block(x, *args, **kwargs)
        return x
        
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
        ) -> Tuple[torch.Tensor, any]:
        mask = self.global_mask_schedule.get_mask()
        if isinstance(self.trg_layers, nn.ModuleList):
            if mask == 1:
                return self.decoder_layers(self.trg_layers, x, *args, **kwargs), None
            if mask == 0:
                return self.decoder_layers(self.meta_layers, x, *args, **kwargs), None
            # meta_output = self.decoder_layers(self.meta_layers, x, *args, **kwargs)
            trg_output = self.decoder_layers(self.trg_layers, x, *args, **kwargs)
            # norm_meta = torch.norm(meta_output, dim=-1)
            # norm_trg = torch.norm(trg_output, dim=-1)
            # product = torch.einsum('ijk,ijk->ij', meta_output, trg_output)
            # cosine_distance = (1 - (product / norm_meta / norm_trg).unsqueeze(-1))
            # # cosine_distance [0, 2]
            # return mask * (1 + cosine_distance - cosine_distance * mask) * trg_output + ((1 - cosine_distance * mask) * (1-mask)) * meta_output, None
            return trg_output, None
        else:
            if mask == 1:
                return self.trg_layers(x, *args, **kwargs)
            return mask * self.trg_layers(x, *args, **kwargs) + (1 - mask) * self.meta_layers(x, *args, **kwargs)


class MilkModel(nn.Module):
    def __init__(self, meta_model, trg_model, config: MilkConfig) -> None:
        super().__init__()
        self.config = config
        self.meta_model = meta_model
        self.trg_model = trg_model
        self.forward = trg_model.forward
        # mask_iters = [0, 10000, 20000, 30000, 40000]
        mask_iters = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
        self.global_mask_schedule = [MaskSchedule(mask_iters[i], mask_iters[i+1]) for i in range(len(mask_iters) - 1)]

        self.find_and_replace()
        del self.meta_model
        torch.cuda.empty_cache()
        self.post_init()
    
    def post_init(self):
        make_only_trg_as_trainable(self.trg_model, self.config.special_modules_mapping)

    def update_special_modules(self):
        meta_key_special_module_mapping = {}

        key_list = [key for key, _ in self.meta_model.named_modules()]
        for key in key_list:
            _, meta_module, meta_module_name = get_submodules(self.meta_model, key)
            if meta_module_name in self.config.special_modules_mapping.values():
                meta_key_special_module_mapping[meta_module_name] = meta_module

        key_list = [key for key, _ in self.trg_model.named_modules()]
        for key in key_list:
            trg_parent, trg_module, trg_module_name = get_submodules(self.trg_model, key)
            if trg_module_name in self.config.special_modules_mapping.keys():
                meta_module_name = self.config.special_modules_mapping[trg_module_name]
                meta_module = meta_key_special_module_mapping[meta_module_name]
                setattr(trg_parent, trg_module_name, copy.deepcopy(meta_module))

    def update_layers(
        self,
        trg_layers_parent,
        trg_layers,
        trg_layers_name,
        meta_layers
    ):
        new_module_list = []
        trg_layers_list = split_layers(trg_layers, self.config.trg_layer_split)
        meta_layers_list = split_layers(meta_layers, self.config.meta_layer_split)
        for layer_index in range(len(trg_layers_list)):
            new_module = IncubationLayer(
                meta_layers=meta_layers_list[layer_index],
                trg_layers=trg_layers_list[layer_index],
                # global_mask_schedule=self.global_mask_schedule,
                global_mask_schedule=self.global_mask_schedule[layer_index]
                )
            new_module_list.append(new_module)
        setattr(trg_layers_parent, trg_layers_name, nn.ModuleList(new_module_list))

    def find_and_replace(self):
        trg_layers_parent, trg_layers, trg_layers_name = find_moduleList(self.trg_model)
        _, meta_layers, _ = find_moduleList(self.meta_model)
        self.update_layers(
            trg_layers_parent,
            trg_layers,
            trg_layers_name,
            meta_layers
            )
        if self.config.special_modules_mapping is not None:
            self.update_special_modules()