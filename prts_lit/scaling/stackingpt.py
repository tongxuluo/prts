import torch
from lit_gpt.model import GPT, Config
from ..utils import (
    StackingptConfig,
    get_partition,
)

class Stackingpt:
    def __init__(self, config) -> None:
        self.config = config
        self.model = None
    
    @classmethod
    def from_pretrained(
        cls,
        config: StackingptConfig,
    ):
        with torch.no_grad():
            stackingpt = cls(config)
            stackingpt.set_param()
        return stackingpt.model

    def set_param(self):
        state_to_load = {}
        src_state = torch.load(self.config.src_init_path)['model']
        for k, p in src_state.items():
            if self.config.embd_name + '.' in k:
                state_to_load.update({k: p})
            if self.config.ln_name + '.' in k or self.config.head_name + '.' in k:
                state_to_load.update({k: p})
            if get_partition(self.config.layer_name) in k:
                pattern = get_partition(self.config.layer_name)
                for i, layer_idx in enumerate(self.config.stacking_list):
                    layer_id = int(k.split(pattern)[-1].split(".")[0])
                    if layer_id == layer_idx:
                        new_k = k.replace(get_partition(str(layer_id)), get_partition(str(i)))
                        state_to_load.update({new_k: p.clone()})
        del src_state
        self.model = GPT(Config.from_name(self.config.trg_config_name))
        self.model.load_state_dict(state_to_load)

