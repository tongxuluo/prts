from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from ..utils import (
    IncubationConfig,
    get_submodules,
    split_layers,
    copy_init,
    find_moduleList,
    make_only_trg_as_trainable
)
from lit_gpt.utils import get_logger


class IncubationLayer(nn.Module):
    def __init__(
        self,
        meta_layers: Union[nn.ModuleList, nn.Linear, nn.Embedding, None],
        trg_layers: Union[nn.ModuleList, nn.Linear, nn.Embedding],
        layer_idx: int,
        global_W: Optional[nn.Parameter] = None,
        use_fused: bool = False,
        activate_g_op: bool = False,
        activate_f_op: bool = False,
    ) -> None:
        super().__init__()
        self.meta_layers = meta_layers if use_fused else None
        self.trg_layers = trg_layers
        self.layer_idx = layer_idx
        self.global_W = global_W
        self.use_fused = use_fused
        self.activate_g_op = activate_g_op
        self.activate_f_op = activate_f_op
        self.logger = get_logger()

    def g(self, input_hidden: torch.tensor):
        if self.global_W is not None and self.activate_g_op:
            return torch.matmul(input_hidden, torch.inverse(self.global_W).transpose(0, 1))
        else:
            return input_hidden

    def f(self, input_hidden: torch.tensor):
        if self.global_W is not None and self.activate_f_op:
            return torch.matmul(input_hidden, self.global_W.transpose(0, 1))
        else:
            return input_hidden
    
    def decoder_layers(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
        ) -> Tuple[torch.Tensor, any]:
        if self.use_fused:
            meta_x = x
            trg_x = self.g(x)
            for meta_block in self.meta_layers:
                meta_x, *_ = meta_block(meta_x, *args, **kwargs)
            for trg_block in self.trg_layers:
                trg_x, *_ = trg_block(trg_x, *args, **kwargs)
            trg_x = self.f(trg_x)
            norm_meta = torch.norm(meta_x, dim=-1)
            norm_trg = torch.norm(trg_x, dim=-1)
            product = torch.einsum('ijk,ijk->ij', meta_x, trg_x)
            cosine_distance = (product / norm_meta / norm_trg).unsqueeze(-1)
            self.logger.log_metrics({"cosine_distance": torch.mean(cosine_distance)})
            # x = cosine_distance * meta_x + (1 - cosine_distance) * trg_x
            x = trg_x
            return x, None
        else:
            x = self.g(x)
            for block in self.trg_layers:
                x, *_ = block(x, *args, **kwargs)
            return self.f(x), None

    def fused_compute(
        self,
        x:torch.Tensor,
        *args,
        **kwargs
        ):
        meta_x = self.meta_layers(x, *args, **kwargs)
        trg_x = self.f(self.trg_layers(self.g(x), *args, **kwargs))
        norm_meta = torch.norm(meta_x, dim=-1)
        norm_trg = torch.norm(trg_x, dim=-1)
        product = torch.einsum('ijk,ijk->ij', meta_x, trg_x)
        cosine_distance = (product / norm_meta / norm_trg).unsqueeze(-1)
        self.logger.log_metrics({"cosine_distance": torch.mean(cosine_distance)})
        return cosine_distance * meta_x + (1 - cosine_distance) * trg_x
        
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
        ) -> Tuple[torch.Tensor, any]:
        if isinstance(self.trg_layers, nn.ModuleList):
            return self.decoder_layers(x, *args, **kwargs)
        elif self.use_fused:
            return self.fused_compute(x, *args, **kwargs)
        else:
            return self.f(self.trg_layers(self.g(x), *args, **kwargs))


class IncubationModel(nn.Module):
    def __init__(self, meta_model, trg_model, config: IncubationConfig) -> None:
        super().__init__()
        self.config = config
        self.meta_model = meta_model
        self.trg_model = trg_model
        self.forward = meta_model.forward
        self.global_W = None
        if self.config.meta_width != self.config.trg_width:
            self.global_W = nn.Parameter(torch.Tensor(self.config.meta_width, self.config.trg_width))
            nn.init.orthogonal_(self.global_W)
        self.find_and_replace()
        self.trg_model = None
        self.post_init()
    
    def post_init(self):
        make_only_trg_as_trainable(self.meta_model)

    def update_special_modules(self):
        trg_key_special_module_mapping = {}

        key_list = [key for key, _ in self.trg_model.named_modules()]
        for key in key_list:
            _, target_module, target_module_name = get_submodules(self.trg_model, key)
            if target_module_name in self.config.special_modules_mapping.values():
                trg_key_special_module_mapping[target_module_name] = target_module

        key_list = [key for key, _ in self.meta_model.named_modules()]
        for key in key_list:
            meta_parent, meta_module, meta_module_name = get_submodules(self.meta_model, key)
            if meta_module_name in self.config.special_modules_mapping.keys():
                trg_module_name = self.config.special_modules_mapping[meta_module_name]
                trg_module = trg_key_special_module_mapping[trg_module_name]
                new_module = IncubationLayer(
                    meta_layers=meta_module,
                    trg_layers=trg_module,
                    layer_idx=-1,
                    global_W=self.global_W,
                    use_fused=self.config.use_fused,
                    activate_g_op=(trg_module_name == self.config.module_activate_g),
                    activate_f_op=(trg_module_name == self.config.module_activate_f)
                )
                if self.config.special_modules_copy_init:
                    copy_init(meta_module, trg_module, self.config.vocab_size)
                setattr(meta_parent, meta_module_name, new_module)

    def update_layers(
        self,
        meta_layers_parent,
        mata_layers,
        meta_layers_name,
        trg_layers
    ):
        new_module_list = []
        mata_layers_list = split_layers(mata_layers, self.config.meta_layer_split)
        trg_layers_list = split_layers(trg_layers, self.config.trg_layer_split)
        for layer_index in range(len(mata_layers_list)):
            if layer_index == self.config.replace_layers_index:
                new_module = IncubationLayer(
                    meta_layers=mata_layers_list[layer_index],
                    trg_layers=trg_layers_list[layer_index],
                    layer_idx=layer_index,
                    global_W=self.global_W,
                    use_fused=self.config.use_fused,
                    activate_g_op=(layer_index == self.config.module_activate_g),
                    activate_f_op=(layer_index == self.config.module_activate_f)
                    )
                new_module_list.append(new_module)
            else:
                new_module_list.extend(mata_layers_list[layer_index])
        setattr(meta_layers_parent, meta_layers_name, nn.ModuleList(new_module_list))

    def find_and_replace(self):
        if self.config.replace_layers_index is not None:
            meta_layers_parent, mata_layers, meta_layers_name = find_moduleList(self.meta_model)
            _, trg_layers, _ = find_moduleList(self.trg_model)
            self.update_layers(
                meta_layers_parent,
                mata_layers,
                meta_layers_name,
                trg_layers
                )
        if self.config.special_modules_mapping is not None:
            self.update_special_modules()