import torch
import torch.nn as nn
import copy
from lit_gpt.model import GPT, Block
from ..utils import (
    GradualStakcingConfig,
    find_moduleList,
)

class GradualStackingModel(nn.Module):
    def __init__(self, model, config: GradualStakcingConfig) -> None:
        super().__init__()
        self.gs_config = config
        self.curr_stage = 0
        self.model = model
        self.forward = model.forward
    
    @classmethod
    def from_pretrained(
        cls,
        model: GPT,
        config: GradualStakcingConfig,
    ):
        with torch.no_grad():
            gs_model = cls(model, config)
            gs_model.grow()
        return gs_model
    
    def grow(self):
        if self.curr_stage == len(self.gs_config.gs_layers_num) - 1:
            return

        with torch.no_grad():
            _, layers, _ = find_moduleList(self.model)
            trg_layer_num = self.gs_config.gs_layers_num[self.curr_stage + 1]
            now_layer_num = self.gs_config.gs_layers_num[self.curr_stage]
            for i in range(trg_layer_num - now_layer_num):
                new_layer = Block(layers[2 * now_layer_num - trg_layer_num + i].config).cuda()
                new_layer.load_state_dict(layers[2 * now_layer_num - trg_layer_num + i].state_dict())
                layers.append(new_layer)

        self.curr_stage += 1

    def step(self, curr_step):
        if curr_step % self.gs_config.grow_step_interval == 0:
            self.grow()
    
    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)
