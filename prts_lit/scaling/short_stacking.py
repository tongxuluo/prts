import torch
import torch.nn as nn
import copy
from lit_gpt.model import GPT, Block
from ..utils import (
    ShortStakcingConfig,
    find_moduleList,
)

class ShortStackingModel(nn.Module):
    def __init__(self, model, config: ShortStakcingConfig) -> None:
        super().__init__()
        self.gs_config = config
        self.model = model
        self.forward = model.forward
    
    @classmethod
    def from_pretrained(
        cls,
        model: GPT,
        config: ShortStakcingConfig,
    ):
        with torch.no_grad():
            ss_model = cls(model, config)
            ss_model.stack()
        return ss_model
    
    def short(self):
        with torch.no_grad():
            _, layers, _ = find_moduleList(self.model)
            trg_layer_num = self.gs_config.gs_layers_num[self.curr_stage + 1]
            now_layer_num = self.gs_config.gs_layers_num[self.curr_stage]
            for i in range(trg_layer_num - now_layer_num):
                new_layer = Block(layers[2 * now_layer_num - trg_layer_num + i].config).cuda()
                new_layer.load_state_dict(layers[2 * now_layer_num - trg_layer_num + i].state_dict())
                layers.append(new_layer)
    
    def stack(self):
        pass

    def step(self, current_step):
        pass

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)
