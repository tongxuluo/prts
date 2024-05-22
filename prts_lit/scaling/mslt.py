import torch
import torch.nn as nn
import copy
from lit_gpt.model import GPT, Block
from ..utils import (
    MsltConfig,
    find_moduleList,
)

class MsltModel(nn.Module):
    def __init__(self, model, config: MsltConfig) -> None:
        super().__init__()
        self.mslt_config = config
        self.curr_stage = 0
        self.model = model
        self.forward = model.forward
    
    @classmethod
    def from_pretrained(
        cls,
        model: GPT,
        config: MsltConfig,
    ):
        with torch.no_grad():
            mslt_model = cls(model, config)
            mslt_model.grow()
        return mslt_model
    
    def grow(self):
        if self.curr_stage == len(self.mslt_config.mslt_layers_num):
            return
        if self.curr_stage == len(self.mslt_config.mslt_layers_num) - 1:
            for n, p in self.model.named_parameters():
                p.requires_grad = True
            self.curr_stage += 1
            return
        
        for n, p in self.model.named_parameters():
            if 'lm_head' not in n and 'ln_f' not in n:
                p.requires_grad = False
        with torch.no_grad():
            _, layers, _ = find_moduleList(self.model)
            trg_layer_num = self.mslt_config.mslt_layers_num[self.curr_stage + 1]
            now_layer_num = self.mslt_config.mslt_layers_num[self.curr_stage]
            for i in range(trg_layer_num - now_layer_num):
                new_layer = Block(layers[2 * now_layer_num - trg_layer_num + i].config).cuda()
                new_layer.load_state_dict(layers[2 * now_layer_num - trg_layer_num + i].state_dict())
                layers.append(new_layer)

        self.curr_stage += 1

    def step(self, curr_step):
        if curr_step % self.mslt_config.grow_step_interval == 0:
            self.grow()
    
    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)
