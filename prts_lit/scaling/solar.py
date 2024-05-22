from typing import Optional, Tuple, Union
import torch
import copy
import torch.nn as nn
from lit_gpt.model import GPT, Config
from ..utils import (
    SolarConfig,
    find_moduleList
)


class SolarModel(nn.Module):
    def __init__(self, src_model, config: SolarConfig) -> None:
        super().__init__()
        self.config = config
        self.src_model = src_model
    
    @classmethod
    def from_pretrained(
        cls,
        model: GPT,
        config: SolarConfig,
    ):
        soalr = cls(model, config)
        soalr.find_and_replace()
        new_model = GPT(Config.from_name(soalr.config.trg_config_name))
        new_model.load_state_dict(soalr.src_model.state_dict())
        del soalr.src_model
        soalr.src_model = new_model
        torch.cuda.empty_cache()
        soalr.src_model.train()
        return soalr.src_model
    
    def expend_depth(
        self,
        layers_parent,
        layers,
        layers_name
    ):
        last_n_modules = layers[-self.config.num_drop_layers:]
        pre_modules = layers[:self.config.src_depth-self.config.num_drop_layers]
        mid_modules = []
        for i in range(self.config.trg_depth - self.config.src_depth):
            mid_modules.append(copy.deepcopy(pre_modules[self.config.num_drop_layers + i]))
        mid_modules = nn.ModuleList(mid_modules)
        new_layers = pre_modules + mid_modules + last_n_modules
        setattr(layers_parent, layers_name, new_layers)

    def find_and_replace(self):
        layers_parent, layers, layers_name = find_moduleList(self.src_model)
        assert layers is not None, "cannot find ModuleList in model"
        self.expend_depth(layers_parent, layers, layers_name)