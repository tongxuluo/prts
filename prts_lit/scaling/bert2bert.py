import random
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from lit_gpt.model import GPT, Config
from lit_gpt.rmsnorm import RMSNorm
from lit_gpt.rmsnorm import FusedRMSNorm
from jsonargparse.cli import CLI
from ..utils import (
    bert2BERTConfig,
    expand_tensor,
    get_submodules,
    find_moduleList
)


class bert2BERT:
    def __init__(self, model, config) -> None:
        self.config = config
        self.model = model
        self.SRC_TO_TRG_MAPPING = {
            config.src_width: config.trg_width,
            3*config.src_width: 3*config.trg_width,
            config.src_depth: config.trg_depth,
            config.src_intermediate_size: config.trg_intermediate_size,
        }

    @classmethod
    def from_pretrained(
        cls,
        model: GPT,
        config: bert2BERTConfig,
    ):
        with torch.no_grad():
            bert2bert = cls(model, config)
            bert2bert.find_and_replace()
            new_state = bert2bert.model.state_dict()
            new_model = GPT(Config.from_name(bert2bert.config.trg_config_name))
            new_model.load_state_dict(new_state)
            del bert2bert.model
            torch.cuda.empty_cache()
            bert2bert.model = new_model
        return bert2bert.model
    
    def get_extra_src_indices(self):
        if self.config.src_width == self.config.trg_width:
            return None
        if self.config.from_left:
            # [0, 1, 2, ..., n_dim - o_dim]
            diff = self.config.trg_width - self.config.src_width
            indices_list = list(range(diff))
            indices_list = [x % self.config.src_width for x in indices_list]
            return_dict = {
                diff : torch.tensor(indices_list),
                3 * diff: torch.tensor(3 * indices_list),
            }
            if self.config.trg_intermediate_size - self.config.src_intermediate_size not in return_dict.keys():
                intermediate_diff = self.config.trg_intermediate_size - self.config.src_intermediate_size
                intermediate_indices_list = (indices_list * 4)[:intermediate_diff]
                return_dict.update({intermediate_diff: torch.tensor(intermediate_indices_list)})
            return return_dict
        else:
            diff = self.config.trg_width - self.config.src_width
            random_indices = random.choices(range(diff), k=diff)
            random_indices = [x % self.config.src_width for x in random_indices]
            return_dict = {
                diff: torch.tensor(random_indices),
                3 * diff: torch.tensor(random_indices * 3),
                4 * diff: torch.tensor(random_indices * 4),
            }
            if self.config.trg_intermediate_size - self.config.src_intermediate_size not in return_dict.keys():
                intermediate_diff = self.config.trg_intermediate_size - self.config.src_intermediate_size
                intermediate_random_indices_list = (random_indices * 4)[:intermediate_diff]
                return_dict.update({intermediate_diff: torch.tensor(intermediate_random_indices_list)})
            return return_dict

    def update_embedding(
        self,
        parent,
        target,
        target_name,
        extra_src_indices,
        device
    ):
        weight = target.weight
        in_dim, out_dim = weight.shape
        new_layer = nn.Embedding(in_dim, self.SRC_TO_TRG_MAPPING[out_dim], padding_idx=target.padding_idx)
        if out_dim in self.SRC_TO_TRG_MAPPING.keys():
            trg_shape = torch.Size((in_dim, self.SRC_TO_TRG_MAPPING[out_dim]))
            weight = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=trg_shape,
                extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[out_dim] - out_dim],
                div=False,
                device=device
            )
        new_layer.weight.data = weight
        setattr(parent, target_name, new_layer)
    
    def update_linear(
        self,
        parent,
        target,
        target_name,
        extra_src_indices,
        device
    ):
        weight = target.weight
        bias = None
        out_dim, in_dim = weight.shape
        in_features = self.SRC_TO_TRG_MAPPING[in_dim] if in_dim in self.SRC_TO_TRG_MAPPING.keys() else in_dim
        out_features = self.SRC_TO_TRG_MAPPING[out_dim] if out_dim in self.SRC_TO_TRG_MAPPING.keys() else out_dim
        
        if target.bias is not None:
            new_layer = nn.Linear(in_features, out_features)
            if out_dim in self.SRC_TO_TRG_MAPPING.keys():
                bias = expand_tensor(
                    tensor=target.bias,
                    trg_shape=torch.Size((self.SRC_TO_TRG_MAPPING[out_dim],)),
                    extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[out_dim] - out_dim],
                    # ffn_extra_src_indices if is_ffn else extra_src_indices,
                    div=False,
                    device=device
                )
            else:
                bias = target.bias
            new_layer.bias.data = bias
        else:
            new_layer = nn.Linear(in_features, out_features, bias=False)
        
        if in_dim in self.SRC_TO_TRG_MAPPING.keys():
            trg_shape = torch.Size((out_dim, self.SRC_TO_TRG_MAPPING[in_dim]))
            weight = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=trg_shape,
                extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[in_dim] - in_dim],
                div=True,
                device=device
            )
        
        out_dim, in_dim = weight.shape

        # if out_dim == self.config.src_width * 3:
        #     # is qkv
        #     q = weight[:self.config.src_width, :]

        #     k = weight[self.config.src_width:self.config.src_width * 2, :]
        #     v = weight[self.config.src_width * 2:, :]
        #     trg_shape = torch.Size((self.SRC_TO_TRG_MAPPING[self.config.src_width], in_dim))
        #     q = expand_tensor(
        #         tensor=q.data.clone(),
        #         trg_shape=trg_shape,
        #         extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[self.config.src_width] - self.config.src_width],
        #         div=False,
        #         device=device
        #     )
        #     k = expand_tensor(
        #         tensor=k.data.clone(),
        #         trg_shape=trg_shape,
        #         extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[self.config.src_width] - self.config.src_width],
        #         div=False,
        #         device=device
        #     )
        #     v = expand_tensor(
        #         tensor=v.data.clone(),
        #         trg_shape=trg_shape,
        #         extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[self.config.src_width] - self.config.src_width],
        #         div=False,
        #         device=device
        #     )
        #     weight = torch.cat((q, k, v), dim=0).contiguous()
        if out_dim in self.SRC_TO_TRG_MAPPING.keys():
            trg_shape = torch.Size((self.SRC_TO_TRG_MAPPING[out_dim], in_dim))
            weight = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=trg_shape,
                extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[out_dim] - out_dim],
                div=False,
                device=device
            )

        new_layer.weight.data = weight
        setattr(parent, target_name, new_layer)

    def update_layernorm(
        self,
        parent,
        target,
        target_name,
        extra_src_indices,
        device
    ):
        weight = target.weight
        bias = target.bias
        out_dim = weight.shape[0]
        new_layer = nn.LayerNorm(self.SRC_TO_TRG_MAPPING[out_dim], eps=target.eps)
        if out_dim in self.SRC_TO_TRG_MAPPING.keys():
            new_layer.weight.data = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=torch.Size((self.SRC_TO_TRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[out_dim] - out_dim],
                div=False,
                device=device
            )
            new_layer.bias.data = expand_tensor(
                tensor=bias.data.clone(),
                trg_shape=torch.Size((self.SRC_TO_TRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[out_dim] - out_dim],
                div=False,
                device=device
            )
        setattr(parent, target_name, new_layer)

    def update_llamaRMSnorm(
        self,
        parent,
        target,
        target_name,
        extra_src_indices,
        device
    ):
        weight = target.weight
        out_dim = weight.shape[0]
        LlamaRMSNorm = type(target)
        new_layer = LlamaRMSNorm(self.SRC_TO_TRG_MAPPING[out_dim], eps=target.eps)
        if out_dim in self.SRC_TO_TRG_MAPPING.keys():
            new_layer.weight.data = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=torch.Size((self.SRC_TO_TRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices[self.SRC_TO_TRG_MAPPING[out_dim] - out_dim],
                div=False,
                device=device
            )
        setattr(parent, target_name, new_layer)

    def expend_layer(
        self,
        extra_src_indices: Optional[Dict],
        device='cuda:0'
        ):
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            parent, target, target_name = get_submodules(self.model, key)
            if isinstance(target, nn.Embedding):
                self.update_embedding(
                    parent,
                    target,
                    target_name,
                    extra_src_indices,
                    device
                )
            elif isinstance(target, nn.Linear):
                self.update_linear(
                    parent,
                    target,
                    target_name,
                    extra_src_indices,
                    device
                )
            elif isinstance(target, nn.LayerNorm):
                self.update_layernorm(
                    parent,
                    target,
                    target_name,
                    extra_src_indices,
                    device
                )
            elif isinstance(target, Union[RMSNorm, FusedRMSNorm]):
                self.update_llamaRMSnorm(
                    parent,
                    target,
                    target_name,
                    extra_src_indices,
                    device
                )

    def expend_depth(
        self,
        layers_parent,
        layers,
        layers_name
    ):
        if self.config.inner_stacking:
            new_layers = []
            for i in range(self.config.trg_depth):
                index = i // (self.config.trg_depth // self.config.src_depth)
                new_layers.append(layers[index])
            new_layers = nn.ModuleList(new_layers)
            setattr(layers_parent, layers_name, new_layers)
        else:
            for i in range(self.config.trg_depth - self.config.src_depth):
                layers.append(layers[i % (self.config.src_depth)])

    def find_and_replace(self):
        extra_src_indices = self.get_extra_src_indices()
        if extra_src_indices is not None:
            self.expend_layer(extra_src_indices)

        layers_parent, layers, layers_name = find_moduleList(self.model)
        assert layers is not None, "cannot find ModuleList in model"
        self.expend_depth(layers_parent, layers, layers_name)


# def test(
#     src_path,
#     model
# ):
#     tokenizer =  AutoTokenizer.from_pretrained(src_path)
#     inputs = tokenizer("Hello world", return_tensors="pt")
#     with torch.no_grad():
#         output = model(**inputs)
#         # print(output)


# def main(
#     src_path: str = "../PTM/uncased_L-6_H-512_A-8",
#     trg_path: str = "../PTM/expend112233_L-12_H-512_A-8",
#     CausalLM: bool = True,
#     MaskedLM: bool = False,
#     inner_stacking: bool = False
# ):
#     src_config = AutoConfig.from_pretrained(src_path)
#     trg_config = AutoConfig.from_pretrained(trg_path)
#     if CausalLM:
#         model = AutoModelForCausalLM.from_pretrained(src_path)
#     elif MaskedLM:
#         model = AutoModelForMaskedLM.from_pretrained(src_path)
#     else:
#         model = AutoModel.from_pretrained(src_path)
#     print(model)
#     bert2bert_config = bert2BERTConfig(
#         src_config,
#         trg_config,
#         False,
#         inner_stacking
#     )
#     bert2bert_model = bert2BERT.from_pretrained(model, bert2bert_config, CausalLM, MaskedLM)
#     print(bert2bert_model)
#     test(src_path, bert2bert_model)
#     bert2bert_model.save_pretrained(trg_path)

# if __name__ == "__main__":
#     CLI(main)