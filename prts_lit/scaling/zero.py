import torch
from lit_gpt.model import GPT
from ..utils import (
    ZeroConfig,
)


class ZeroModel:
    def __init__(self, model, config: ZeroConfig) -> None:
        self.model = model
        self.msg_config = config
        self.grow()
    
    def expand_tensor(self, src_tensor, trg_tensor):
        if len(src_tensor.size()) == 2:
            dout, din = src_tensor.size()
            trg_tensor[:dout, :din] = src_tensor
            trg_tensor[dout:, :din] = 0
        else:
            dout = src_tensor.size()[0]
            trg_tensor[:dout] = src_tensor
            trg_tensor[dout:] = 0
        return trg_tensor

    def grow(self):
        state_dict = torch.load(self.msg_config.src_path)['model']
        for k, p in self.model.named_parameters():
            if k in state_dict:
                src_tensors = state_dict[k]
                if 'wte' in k:
                    p.data = self.expand_tensor(src_tensors.T, p.data.T).T
                else:
                    p.data = self.expand_tensor(src_tensors, p.data)
        state_dict = None
    
    @classmethod
    def from_pretrained(
        cls,
        model: GPT,
        config: ZeroConfig,
    ):
        with torch.no_grad():
            zero_model = cls(model, config)
        return zero_model.model

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)