import random
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from transformers import PreTrainedModel, AutoModelForMaskedLM
import torch.nn as nn
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
        self.SRC_TO_RRG_MAPPING = {
            config.src_width: config.trg_width,
            config.src_depth: config.trg_depth,
            config.src_intermediate_size: config.trg_intermediate_size,
        }
        if self.model.config.model_type == "gpt_neox":
            self.SRC_TO_RRG_MAPPING.update({
                3 * config.src_width: 3 * config.trg_width
            })

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        config: bert2BERTConfig,
        CausalLM: bool = True,
        MaskedLM: bool = False,
    ):
        bert2bert = cls(model, config)
        bert2bert.find_and_replace()
        if CausalLM:
            new_model = AutoModelForCausalLM.from_config(bert2bert.config.trg_config)
        elif MaskedLM:
            new_model = AutoModelForMaskedLM.from_config(bert2bert.config.trg_config)
        else:
            new_model = AutoModel.from_config(bert2bert.config.trg_config)
        new_model.load_state_dict(bert2bert.model.state_dict())
        bert2bert.model = new_model
        return bert2bert.model
    
    def get_extra_src_indices(self):
        if self.config.src_width == self.config.trg_width:
            return None
        if self.config.from_left:
            # [0, 1, 2, ..., n_dim - o_dim]
            diff = self.config.trg_width - self.config.src_width
            indices_list = range(diff)
            return {
                diff : torch.tensor(indices_list),
                3 * diff: torch.tensor(3 * indices_list),
                4 * diff: torch.tensor(4 * indices_list),
            }
        else:
            # 从 [0, 1, 2, ..., n_dim - o_dim] 随机选取
            diff = self.config.trg_width - self.config.src_width

            random_indices = random.choices(range(diff), k=diff)
            return {
                diff: torch.tensor(random_indices),
                3 * diff: torch.tensor(random_indices * 3),
                4 * diff: torch.tensor(random_indices * 4),
            }

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
        new_layer = nn.Embedding(in_dim, self.SRC_TO_RRG_MAPPING[out_dim], padding_idx=target.padding_idx)
        if out_dim in self.SRC_TO_RRG_MAPPING.keys():
            trg_shape = torch.Size((in_dim, self.SRC_TO_RRG_MAPPING[out_dim]))
            weight = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=trg_shape,
                extra_src_indices=extra_src_indices[self.SRC_TO_RRG_MAPPING[out_dim] - out_dim],
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
        in_features = self.SRC_TO_RRG_MAPPING[in_dim] if in_dim in self.SRC_TO_RRG_MAPPING.keys() else in_dim
        out_features = self.SRC_TO_RRG_MAPPING[out_dim] if out_dim in self.SRC_TO_RRG_MAPPING.keys() else out_dim
        
        if target.bias is not None:
            new_layer = nn.Linear(in_features, out_features)
            if out_dim in self.SRC_TO_RRG_MAPPING.keys():
                bias = expand_tensor(
                    tensor=target.bias,
                    trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                    extra_src_indices=extra_src_indices[self.SRC_TO_RRG_MAPPING[out_dim] - out_dim],
                    # ffn_extra_src_indices if is_ffn else extra_src_indices,
                    div=False,
                    device=device
                )
            else:
                bias = target.bias
            new_layer.bias.data = bias
        else:
            new_layer = nn.Linear(in_features, out_features, bias=False)
        
        if out_dim in self.SRC_TO_RRG_MAPPING.keys():
            trg_shape = torch.Size((self.SRC_TO_RRG_MAPPING[out_dim], in_dim))
            weight = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=trg_shape,
                extra_src_indices=extra_src_indices[self.SRC_TO_RRG_MAPPING[out_dim] - out_dim],
                # ffn_extra_src_indices if up_ffn else extra_src_indices,
                div=True,
                device=device
            )
        out_dim, in_dim = weight.shape
        if in_dim in self.SRC_TO_RRG_MAPPING.keys():
            trg_shape = torch.Size((out_dim, self.SRC_TO_RRG_MAPPING[in_dim]))
            weight = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=trg_shape,
                extra_src_indices=extra_src_indices[self.SRC_TO_RRG_MAPPING[in_dim] - in_dim],
                # ffn_extra_src_indices if down_ffn else extra_src_indices,
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
        new_layer = nn.LayerNorm(self.SRC_TO_RRG_MAPPING[out_dim], eps=target.eps)
        if out_dim in self.SRC_TO_RRG_MAPPING.keys():
            new_layer.weight.data = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices[self.SRC_TO_RRG_MAPPING[out_dim] - out_dim],
                div=False,
                device=device
            )
            new_layer.bias.data = expand_tensor(
                tensor=bias.data.clone(),
                trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices[self.SRC_TO_RRG_MAPPING[out_dim] - out_dim],
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
        new_layer = LlamaRMSNorm(self.SRC_TO_RRG_MAPPING[out_dim], eps=target.variance_epsilon)
        if out_dim in self.SRC_TO_RRG_MAPPING.keys():
            new_layer.weight.data = expand_tensor(
                tensor=weight.data.clone(),
                trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices[self.SRC_TO_RRG_MAPPING[out_dim] - out_dim],
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
            parent, target, target_name = get_submodules(key)
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
            elif type(target).__name__ == "LlamaRMSNorm":
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

        layers_parent, layers, layers_name = find_moduleList()
        assert layers is not None, "cannot find ModuleList in model"
        self.expend_depth(layers_parent, layers, layers_name)


def test(
    src_path,
    model
):
    tokenizer =  AutoTokenizer.from_pretrained(src_path)
    inputs = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        # print(output)


def main(
    src_path: str = "../PTM/uncased_L-6_H-512_A-8",
    trg_path: str = "../PTM/expend112233_L-12_H-512_A-8",
    CausalLM: bool = True,
    MaskedLM: bool = False,
    inner_stacking: bool = False
):
    src_config = AutoConfig.from_pretrained(src_path)
    trg_config = AutoConfig.from_pretrained(trg_path)
    if CausalLM:
        model = AutoModelForCausalLM.from_pretrained(src_path)
    elif MaskedLM:
        model = AutoModelForMaskedLM.from_pretrained(src_path)
    else:
        model = AutoModel.from_pretrained(src_path)
    print(model)
    bert2bert_config = bert2BERTConfig(
        src_config,
        trg_config,
        False,
        inner_stacking
    )
    bert2bert_model = bert2BERT.from_pretrained(model, bert2bert_config, CausalLM, MaskedLM)
    print(bert2bert_model)
    test(src_path, bert2bert_model)
    bert2bert_model.save_pretrained(trg_path)

if __name__ == "__main__":
    CLI(main)