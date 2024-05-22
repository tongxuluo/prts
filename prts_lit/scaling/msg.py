import math
from typing import Tuple, Union
import torch
import torch.nn as nn
from lit_gpt.model import GPT, Block
from lit_gpt.rmsnorm import RMSNorm
from lit_gpt.rmsnorm import FusedRMSNorm
from ..utils import (
    MsgConfig,
    get_submodules,
    find_moduleList
)


class MsgMask:
    def __init__(
        self,
        src_width,
        trg_width,
        src_intermediate_size,
        trg_intermediate_size,
        grow_step
    ) -> None:
        self.curr_step = 0
        self.grow_step = grow_step
        self.src_width = src_width
        self.trg_width = trg_width
        self.src_intermediate_size = src_intermediate_size
        self.trg_intermediate_size = trg_intermediate_size
    
    def step(self):
        self.curr_step += 1

    def getfullmask(self, dtype, device):
        if self.curr_step < self.grow_step:
            mask = torch.zeros(1, 1, self.trg_width, dtype=dtype, device=device)
            mask[:, :, :] = self.curr_step / self.grow_step
            return mask
        else:
            mask = torch.ones(1, 1, self.trg_width, dtype=dtype, device=device)
            return mask
    
    def getmask(self, dim, dtype, device):
        mask = torch.zeros(1, 1, dim, dtype=dtype, device=device)
        if self.curr_step < self.grow_step:
            if dim == 3 * self.trg_width:
                mask = torch.ones(1, 1, dim, dtype=dtype, device=device)
                mask[:, :, 3*self.src_width:] = 0
            elif dim == self.trg_intermediate_size:
                mask[:, :, :self.src_intermediate_size] = 1
                mask[:, :, self.src_intermediate_size:] = self.curr_step / self.grow_step
            elif dim == self.trg_width:
                mask[:, :, :self.src_width] = 1
                mask[:, :, self.src_width:] = 0
            else:
                mask = None
        elif self.curr_step < 2 * self.grow_step:
            if dim == 3 * self.trg_width:
                mask = torch.ones(1, 1, dim, dtype=dtype, device=device)
                mask[:, :, 3*self.src_width:] = 0
            elif dim == self.trg_intermediate_size:
                mask[:, :, :self.src_intermediate_size] = 1
                mask[:, :, self.src_intermediate_size:] = 1
            elif dim == self.trg_width:
                mask[:, :, :self.src_width] = 1
                mask[:, :, self.src_width:] = (self.curr_step - self.grow_step) / self.grow_step
            else:
                mask = None
        elif self.curr_step < 3 * self.grow_step:
            if dim == 3 * self.trg_width:
                mask = torch.ones(1, 1, dim, dtype=dtype, device=device)
                mask[:, :, 3*self.src_width:] = (self.curr_step - 2*self.grow_step) / self.grow_step
            elif dim == self.trg_intermediate_size:
                mask[:, :, :self.src_intermediate_size] = 1
                mask[:, :, self.src_intermediate_size:] = 1
            elif dim == self.trg_width:
                mask[:, :, :self.src_width] = 1
                mask[:, :, self.src_width:] = 1
            else:
                mask = None
        else:
            mask = torch.ones(1, 1, dim, dtype=dtype, device=device)
        return mask


class MsgNorm(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        mask,
    ) -> None:
        super().__init__()
        self.module = module
        self.mask = mask
    
    def forward(self, x):
        dim = x.size()[-1]
        mask = self.mask.getmask(dim, x.dtype, x.device)
        y = self.module(x * mask) / math.sqrt(dim) * math.sqrt(mask.sum())
        return y * mask


class MsgLayer(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        mask,
    ) -> None:
        super().__init__()
        self.module = module
        self.mask = mask
    
    def forward(self, x):
        y = self.module(x)
        dim = y.size()[-1]
        mask = self.mask.getmask(dim, y.dtype, y.device)
        return y * mask

class MsgBlock(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        mask,
    ) -> None:
        super().__init__()
        self.module = module
        self.mask = mask
    
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
        ) -> Tuple[torch.Tensor, any]:
        y, _ = self.module(x, *args, **kwargs)
        mask = self.mask.getfullmask(y.dtype, y.device)
        return y * mask + x * (1-mask), _

class MsgModel(nn.Module):
    def __init__(self, model, config: MsgConfig) -> None:
        super().__init__()
        self.model = model
        self.msg_config = config
        self.forward = model.forward 
        self.mask = MsgMask(config.src_width, config.trg_width, config.src_intermediate_size, config.trg_intermediate_size, config.grow_step)
        self.grow()
        self.find_and_replace()
        print(self.model)
    
    def expand_tensor(self, src_tensor, trg_tensor):
        if len(src_tensor.size()) == 2:
            dout, din = src_tensor.size()
            trg_tensor[:dout, :din] = src_tensor
        else:
            dout = src_tensor.size()[0]
            trg_tensor[:dout] = src_tensor
        return trg_tensor

    def grow(self):
        state_dict = torch.load(self.msg_config.src_path)['model']
        for k, p in self.model.named_parameters():
            if k in state_dict:
                src_tensors = state_dict[k]
            # else:
            #     layer_id = int(k.split('.h.')[-1].split(".")[0])
            #     new_k = k.replace(f'.{layer_id}.', f'.{layer_id % self.msg_config.src_depth}.')
            #     src_tensors = state_dict[new_k]
                p.data = self.expand_tensor(src_tensors, p.data)
        state_dict = None
    
    def find_and_replace(self):
        key_list = [key for key, _ in self.model.named_modules()]
        if self.msg_config.src_width != self.msg_config.trg_width:
            for key in key_list:
                if 'lm_head' in key:
                    continue
                parent, target, target_name = get_submodules(self.model, key)
                if isinstance(target, Union[nn.Linear, nn.Embedding]):
                    new_module = MsgLayer(target, self.mask)
                    setattr(parent, target_name, new_module)
                elif isinstance(target, Union[nn.LayerNorm, RMSNorm, FusedRMSNorm]):
                    new_module = MsgNorm(target, self.mask)
                    setattr(parent, target_name, new_module)
        
        if self.msg_config.src_depth != self.msg_config.trg_depth:
            _, layers, _ = find_moduleList(self.model)
            for i in range(self.msg_config.src_depth, self.msg_config.trg_depth):
                new_module = MsgBlock(layers[i], self.mask)
                layers[i] = new_module

    
    def step(self):
        self.mask.step()
    
    @classmethod
    def from_pretrained(
        cls,
        model: GPT,
        config: MsgConfig,
    ):
        with torch.no_grad():
            msg_model = cls(model, config)
        return msg_model

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)