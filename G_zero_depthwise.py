import copy
import math
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForMaskedLM
from dataclasses import dataclass
import torch.nn as nn
from jsonargparse.cli import CLI

@dataclass
class bert2BERTConfig:
    src_config: PretrainedConfig
    trg_config: PretrainedConfig
    from_left: bool
    lemon: bool

    def __post_init__(self):
        self.vocab_size = self.src_config.vocab_size
        self.src_width = self.src_config.hidden_size
        self.src_depth = self.src_config.num_hidden_layers
        self.src_intermediate_size = self.src_config.intermediate_size
        self.trg_width = self.trg_config.hidden_size
        self.trg_depth = self.trg_config.num_hidden_layers
        self.trg_intermediate_size = self.trg_config.intermediate_size


class bert2BERT:
    def __init__(self, model, config) -> None:
        self.config = config
        self.model = model
        self.SRC_TO_RRG_MAPPING = {
            config.src_width: config.trg_width,
            config.src_depth: config.trg_depth,
            config.src_intermediate_size: config.trg_intermediate_size
        }

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
            return torch.tensor(range(self.config.trg_width - self.config.from_left))
        else:
            # 从 [0, 1, 2, ..., n_dim - o_dim] 随机选取
            return torch.randint(0, self.config.trg_width - self.config.src_width, size=(self.config.trg_width - self.config.src_width,))

    def make_ffn_extra_src_indices(self, extra_src_indices):
        ffn_size = self.config.trg_intermediate_size - self.config.src_intermediate_size
        repeated_index = extra_src_indices.repeat((ffn_size // len(extra_src_indices)) + 1)
        ffn_index = repeated_index[:ffn_size]
        return ffn_index

    def expand_tensor(
        self,
        tensor: torch.Tensor,  # the tensor needing expansion (a, b)
        trg_shape: torch.Size,  
        extra_src_indices: Optional[torch.Tensor],
        div=True,
        lemon=False,
        device='cuda:0' 
    ):
        if extra_src_indices is None:
            return tensor
        assert len(tensor.shape) == len(trg_shape)
        assert len(extra_src_indices.shape) == 1
        
        extra_src_tensor = tensor
        non_align_dim = [dim for dim in range(len(tensor.shape)) if tensor.size(dim) != trg_shape[dim]]

        assert len(non_align_dim) <= 1, 'only support one dimension expansion'

        dim = non_align_dim[0]

        # convert device
        origin_device = tensor.device
        tensor = tensor.to(device)
        extra_src_tensor = extra_src_tensor.to(device)
        extra_src_indices = extra_src_indices.to(device)

        # rand初始化一个new_tensor
        new_tensor = torch.randn(trg_shape, dtype=tensor.dtype, device=device)
        # 赋值原来的部分
        new_tensor.narrow(dim, 0, tensor.size(dim)).copy_(tensor)
        padding_tensor = extra_src_tensor.index_select(dim, extra_src_indices)

        if div:
            # count the repeated times of each element in extra_src_indices
            repeat_count = (extra_src_indices.unsqueeze(-1) == extra_src_indices).sum(dim=-1)
            for _ in range(0, dim):
                repeat_count = repeat_count.unsqueeze(0)
            for _ in range(dim+1, len(padding_tensor.shape)):
                repeat_count = repeat_count.unsqueeze(-1)
            
            if not torch.all(tensor.index_select(dim, extra_src_indices)==padding_tensor):
                print('Warning: div=True and the source of paddings is not tensor, the result may be unexpected')
                # print(tensor.index_select(dim, extra_src_indices)==padding_tensor)
                padding_tensor = padding_tensor / repeat_count
            elif lemon:
                random_weight = 1 / (repeat_count+1) + (torch.randn(padding_tensor.shape, device=device) * (1 / math.sqrt(trg_shape[-1]) / self.config.trg_depth))
                unique_elements, counts = torch.unique(extra_src_indices, return_counts=True)
                old_weight = 1 - random_weight @ (unique_elements.unsqueeze(0) == extra_src_indices.unsqueeze(-1)).to(random_weight.dtype)
                old_tensor = extra_src_tensor.index_select(dim, unique_elements) * old_weight
                padding_tensor = padding_tensor * random_weight
                new_tensor.index_copy_(dim, unique_elements, old_tensor)
            else:
                repeat_count = repeat_count + 1 # +的这个1是复制的那一份
                padding_tensor = padding_tensor / repeat_count
                # 原来的参数位置也需要除以repeat_count
                new_tensor.index_copy_(dim, extra_src_indices, padding_tensor)
        # print(new_tensor.shape)
        # print(padding_tensor.shape)
        new_tensor.narrow(dim,tensor.size(dim), trg_shape[dim]-tensor.size(dim)).copy_(padding_tensor)
        new_tensor = new_tensor.to(origin_device)
        tensor = tensor.to(origin_device)
        extra_src_tensor = extra_src_tensor.to(origin_device)
        extra_src_indices = extra_src_indices.to(origin_device)
        return new_tensor

    def get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

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
            weight = self.expand_tensor(
                tensor=weight,
                trg_shape=trg_shape,
                extra_src_indices=extra_src_indices,
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
        ffn_extra_src_indices = self.make_ffn_extra_src_indices(extra_src_indices)
        up_ffn =False
        down_ffn =False

        weight = target.weight
        bias = None
        out_dim, in_dim = weight.shape
        if out_dim == self.config.src_intermediate_size:
            up_ffn = True
        if in_dim == self.config.src_intermediate_size:
            down_ffn = True
        in_features = self.SRC_TO_RRG_MAPPING[in_dim] if in_dim in self.SRC_TO_RRG_MAPPING.keys() else in_dim
        out_features = self.SRC_TO_RRG_MAPPING[out_dim] if out_dim in self.SRC_TO_RRG_MAPPING.keys() else out_dim
        
        if target.bias is not None:
            new_layer = nn.Linear(in_features, out_features)
            is_ffn =False
            if out_dim == self.config.src_intermediate_size:
                is_ffn = True
            if out_dim in self.SRC_TO_RRG_MAPPING.keys():
                bias = self.expand_tensor(
                    tensor=target.bias,
                    trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                    extra_src_indices=ffn_extra_src_indices if is_ffn else extra_src_indices,
                    div=False,
                    device=device
                )
            else:
                bias = target.bias
            new_layer.bias.data = bias
        else:
            new_layer = nn.Linear(in_features, out_features, bias=False)
        
        if in_dim in self.SRC_TO_RRG_MAPPING.keys():
            trg_shape = torch.Size((out_dim, self.SRC_TO_RRG_MAPPING[in_dim]))
            weight = self.expand_tensor(
                tensor=weight,
                trg_shape=trg_shape,
                extra_src_indices= ffn_extra_src_indices if down_ffn else extra_src_indices,
                div=True,
                lemon=self.config.lemon,
                device=device
            )
        out_dim, in_dim = weight.shape
        if out_dim in self.SRC_TO_RRG_MAPPING.keys():
            trg_shape = torch.Size((self.SRC_TO_RRG_MAPPING[out_dim], in_dim))
            weight = self.expand_tensor(
                tensor=weight,
                trg_shape=trg_shape,
                extra_src_indices=ffn_extra_src_indices if up_ffn else extra_src_indices,
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
            new_layer.weight.data = self.expand_tensor(
                tensor=weight,
                trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices,
                div=False,
                device=device
            )
            new_layer.bias.data = self.expand_tensor(
                tensor=bias,
                trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices,
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
            new_layer.weight.data = self.expand_tensor(
                tensor=weight,
                trg_shape=torch.Size((self.SRC_TO_RRG_MAPPING[out_dim],)),
                extra_src_indices=extra_src_indices,
                div=False,
                device=device
            )
            if self.config.lemon:
                new_layer.weight.data = new_layer.weight.data * out_dim / self.SRC_TO_RRG_MAPPING[out_dim]
        setattr(parent, target_name, new_layer)

    def expend_layer(
        self,
        extra_src_indices: Optional[torch.Tensor],
        device='cuda:0'
        ):
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            parent, target, target_name = self.get_submodules(key)
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
        if self.config.lemon:
            flag = [False] * self.config.src_depth
            new_layers = []
            for i in range(self.config.trg_depth):
                index = i // (self.config.trg_depth // self.config.src_depth)
                if flag[index]:
                    layers[index].self_attn.o_proj.weight.data = torch.zeros_like(layers[index].self_attn.o_proj.weight.data)
                    layers[index].mlp.down_proj.weight.data = torch.zeros_like(layers[index].mlp.down_proj.weight.data)
                    new_layers.append(copy.deepcopy(layers[index]))
                else:
                    new_layers.append(copy.deepcopy(layers[index]))
                    flag[index] = True
            new_layers = nn.ModuleList(new_layers)
            setattr(layers_parent, layers_name, new_layers)
        else:
            for i in range(self.config.trg_depth - self.config.src_depth):
                layers.append(layers[i % (self.config.src_depth)])

    def find_moduleList(self):
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            parent, layer, layer_name = self.get_submodules(key)
            if isinstance(layer, nn.ModuleList):
                return parent, layer, layer_name

    def find_and_replace(self):
        extra_src_indices = self.get_extra_src_indices()
        if extra_src_indices is not None:
            self.expend_layer(extra_src_indices)

        if self.config.src_depth != self.config.trg_depth:
            layers_parent, layers, layers_name = self.find_moduleList()
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
    trg_path: str = "../PTM/expend_L-24_H-512_A-8",
    CausalLM: bool = True,
    MaskedLM: bool = False,
    lemon: bool = False,
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
        lemon,
    )
    bert2bert_model = bert2BERT.from_pretrained(model, bert2bert_config, CausalLM, MaskedLM)
    print(bert2bert_model)
    test(src_path, bert2bert_model)
    bert2bert_model.save_pretrained(trg_path)

if __name__ == "__main__":
    CLI(main)