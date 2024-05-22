from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import math
from higher.patch import make_functional
from collections import OrderedDict
from collections import defaultdict
from ..utils import (
    LiGOConfig,
)


class LiGOModel(nn.Module):
    def __init__(
        self,
        src_model,
        trg_model,
        config: LiGOConfig = None,
    ):
        super().__init__()
        
        self.config = config
        
        # trg model
        self.trg_model = trg_model

        # for model width expansion
        self.B_embd = nn.Parameter(self.init(self.config.trg_width, self.config.src_width, 'width'))
        self.BQ = nn.Parameter(self.init(self.config.trg_width, self.config.src_width, 'width'))
        self.BK = nn.Parameter(self.init(self.config.trg_width, self.config.src_width, 'width'))
        self.BV = nn.Parameter(self.init(self.config.trg_width, self.config.src_width, 'width'))

        # self.BQKV = nn.Parameter(self.init(
        #     self.config.trg_width + 2 * self.config.trg_width * self.config.trg_query_groups // self.config.trg_head,
        #     self.config.src_width + 2 * self.config.src_width * self.config.src_query_groups // self.config.src_head,
        #     'width'
        #     ))

        # src_ffn_dim = find_multiple(int(8 / 3 * self.config.src_width), 256)
        # trg_ffn_dim = find_multiple(int(8 / 3 * self.config.trg_width), 256)
        # self.B_ffn = nn.Parameter(self.init(trg_ffn_dim, src_ffn_dim, 'width'))
        self.B_ffn = nn.Parameter(self.init(self.config.trg_intermediate_size, self.config.src_intermediate_size, 'width'))

        # for model depth expansion
        # for attention fusion
        self.WQ = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        self.WK = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        self.WV = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        self.WO = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        # for layer norm fusion
        self.W_ln1 = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        self.W_ln2 = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        # for ffn fusion
        # self.W_fc1 = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        # self.W_fc2 = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        self.W_fc1 = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        self.W_fc2 = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))
        self.W_proj = nn.Parameter(self.init(self.config.trg_depth, self.config.src_depth, 'depth'))

        # register src state dict as buffer
        self.connect_char = "*"
        for n, p in src_model.state_dict().items():
            self.register_buffer(f'{self.connect_char}'.join(n.split('.')), p)
    
    @classmethod
    def from_pretrained(
        cls,
        src_model,
        trg_model,
        config: LiGOConfig = None,
    ):
        ligo = cls(src_model, trg_model, config)
        return ligo

    def get_hypernet_dict(self, ):
        sd = OrderedDict()
        for n, p in self.named_parameters():
            if 'trg_model' not in n:
                sd[n] = p.data
        return sd

    def init(
        self, 
        dim1, 
        dim2, 
        mode='width'  # choose from width and depth 
    ):
        assert dim1 >= dim2, "right now we just supprt small -> large expansion"
        if mode == 'width':
            out = torch.cat([torch.eye(dim2), torch.randn(dim1 - dim2, dim2) * math.sqrt(1 / dim1)])
        else:
            assert mode == 'depth'
            dims = [dim2 for _ in range(dim1 // dim2)]
            if dim1 % dim2 != 0:
                dims.append(dim1 - sum(dims))
            out = torch.cat([torch.eye(d) for d in dims], dim=0)
        # assert out.shape[0] == dim1 and out.shape[2] == dim2, out.shape
        return out

    def forward(self, x):
        trg_params = self.get_trg_params()
        fmodel = make_functional(self.trg_model).eval()
        # TODO check the prefix of the target model named_parameters()
        tt = [trg_params[n.replace('._fsdp_wrapped_module', '')] for n, _ in self.trg_model.named_parameters()]
        return fmodel(
            x, 
            params=[trg_params[n.replace('._fsdp_wrapped_module', '')] 
                    for n, _ in self.trg_model.named_parameters()]
        )

    def width_expansion(self, ):
        # Width expansion
        params_temp = OrderedDict()
        params_temp_w = defaultdict(list)
        for lnum in range(self.config.src_depth):
            pfx= f'transformer.h.{lnum}'

            # layer norm expansion
            for wn in ['norm_1.weight', 'norm_2.weight']:
                name = f'{pfx}.{wn}'
                buffer_name = f'{self.connect_char}'.join(name.split('.'))
                params_temp[name] = getattr(self, buffer_name) @ self.B_embd.T
                params_temp_w[wn].append(params_temp[name])

            # attention expansion
            # q, k, v attention expansion
            name = f"{pfx}.attn.attn.weight"
            buffer_name = f'{self.connect_char}'.join(name.split('.'))
            q, k, v = torch.split(getattr(self, buffer_name), self.config.src_width, dim=0)
            attn_weights = []
            for w, we in [(q, self.BQ), (k, self.BK), (v, self.BV)]:
                attn_weights.append(we @ w @ self.B_embd.T)
            params_temp[name] = torch.cat(attn_weights, dim=0)

            params_temp_w['attn.attn.weight'].append(params_temp[name])

            # output matrix expansion
            name = f"{pfx}.attn.proj.weight"
            buffer_name = f'{self.connect_char}'.join(name.split('.'))
            params_temp[name] = self.B_embd @ getattr(self, buffer_name) @ self.BV.T
            params_temp_w['attn.proj.weight'].append(params_temp[name])

            # FFN expansion
            # for wn in ['mlp.fc1.weight', 'mlp.fc2.weight']:
            for wn in ['mlp.swiglu.w1.weight', 'mlp.swiglu.w2.weight']:
                name = f'{pfx}.{wn}'
                buffer_name = f'{self.connect_char}'.join(name.split('.'))
                params_temp[name] = self.B_ffn @ getattr(self, buffer_name) @ self.B_embd.T
                params_temp_w[wn].append(params_temp[name])
            
            name = f'{pfx}.mlp.swiglu.w3.weight'
            buffer_name = f'{self.connect_char}'.join(name.split('.'))
            params_temp[name] = self.B_embd @ getattr(self, buffer_name) @ self.B_ffn.T
            params_temp_w['mlp.swiglu.w3.weight'].append(params_temp[name])

            # name = f"{pfx}.mlp.proj.weight"
            # buffer_name = f'{self.connect_char}'.join(name.split('.'))
            # params_temp[name] = self.B_embd @ getattr(self, buffer_name) @ self.B_ffn.T
            # params_temp_w['mlp.proj.weight'].append(params_temp[name])

        return params_temp, params_temp_w

    def get_trg_params(self):
        trg_params = OrderedDict()
        # embedding expansion
        name = 'transformer.wte.weight'
        buffer_name = f'{self.connect_char}'.join(name.split('.'))
        trg_params[name] = getattr(self, buffer_name) @ self.B_embd.T

        # lm_head expansion
        name = 'lm_head.weight'
        buffer_name = f'{self.connect_char}'.join(name.split('.'))
        trg_params[name] = getattr(self, buffer_name) @ self.B_embd.T

        
        # norm (before lm_head) expansion
        name = 'transformer.ln_f.weight'
        buffer_name = f'{self.connect_char}'.join(name.split('.'))
        trg_params[name] = getattr(self, buffer_name) @ self.B_embd.T

        # width expansion
        _, params_temp_w = self.width_expansion()

        # dw for Depth expansion Weights
        """
        param_name_dw = [
            'norm_1.weight', 'norm_2.weight',
            'attn.attn.weight', 'attn.proj.weight',
            'mlp.fc1.weight', 'mlp.fc2.weight', 'mlp.proj.weight'
        ]
        """
        # depth expansion
        for lnum in range(self.config.trg_depth):
            pfx = f'transformer.h.{lnum}'

            # layer norm fusion
            w_name = 'norm_1.weight'
            name = f"{pfx}.{w_name}"
            for i in range(self.config.src_depth):
                trg_params[name] = trg_params.get(name, 0) + self.W_ln1[lnum, i] * params_temp_w[w_name][i]

            w_name = 'norm_2.weight'
            name = f"{pfx}.{w_name}"
            for i in range(self.config.src_depth):
                trg_params[name] = trg_params.get(name, 0) + self.W_ln2[lnum, i] * params_temp_w[w_name][i]
            
            # attention kqv weight fusion
            w_name = 'attn.attn.weight'
            name = f"{pfx}.{w_name}"
            q_sum, k_sum, v_sum = 0, 0, 0
            for i in range(self.config.src_depth):
                c_attn = params_temp_w[w_name][i]
                q, k, v = torch.split(c_attn, self.config.trg_width, dim=0)
                q_sum = q_sum + self.WQ[lnum, i] * q
                k_sum = k_sum + self.WK[lnum, i] * k
                v_sum = v_sum + self.WV[lnum, i] * v
            trg_params[name] = torch.cat([q_sum, k_sum, v_sum], dim=0)

            # attnetion output_proj weight fusion
            w_name = 'attn.proj.weight'
            name = f"{pfx}.{w_name}"
            for i in range(self.config.src_depth):
                trg_params[name] = trg_params.get(name, 0) + self.WO[lnum, i] * params_temp_w[w_name][i]

            # FFN fusion
            w_name_dict = {
                'mlp.swiglu.w1.weight': self.W_fc1,
                'mlp.swiglu.w2.weight': self.W_fc2,
                'mlp.swiglu.w3.weight': self.W_proj
            }
            for w_name, dm in w_name_dict.items():
                name = f"{pfx}.{w_name}"
                for i in range(self.config.src_depth):
                    trg_params[name] = trg_params.get(name, 0) + dm[lnum, i] * params_temp_w[w_name][i]
        
        return trg_params