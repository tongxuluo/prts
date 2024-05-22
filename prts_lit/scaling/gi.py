import torch.nn as nn
from ..utils import (
    split_layers,
    make_only_trg_as_trainable,
    make_only_before_n_layer_trg_as_trainable
)
from .incubation import IncubationLayer, IncubationModel


class GradualIncubation(IncubationModel):
    def post_init(self):
        make_only_trg_as_trainable(self.meta_model)
        if self.config.activate_before_n_layer is not None:
            make_only_before_n_layer_trg_as_trainable(
                self.meta_model,
                self.config.activate_before_n_layer,
            )

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
            if layer_index <= self.config.replace_layers_index:
                new_module = IncubationLayer(
                    meta_layers=mata_layers_list[layer_index],
                    trg_layers=trg_layers_list[layer_index],
                    layer_idx=layer_index,
                    global_W=self.global_W,
                    activate_f_op=(layer_index == self.config.module_activate_f)
                    )
                new_module_list.append(new_module)
            else:
                new_module_list.extend(mata_layers_list[layer_index])
        setattr(meta_layers_parent, meta_layers_name, nn.ModuleList(new_module_list))
    
    def step():
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