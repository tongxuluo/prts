from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from transformers import PretrainedConfig
from lit_gpt.model import Config


@dataclass
class bert2BERTConfig:
    src_config: PretrainedConfig
    trg_config: PretrainedConfig
    from_left: bool
    inner_stacking: bool

    def __post_init__(self):
        self.vocab_size = self.src_config.vocab_size
        self.src_width = self.src_config.hidden_size
        self.src_depth = self.src_config.num_hidden_layers
        self.src_intermediate_size = self.src_config.intermediate_size
        self.trg_width = self.trg_config.hidden_size
        self.trg_depth = self.trg_config.num_hidden_layers
        self.trg_intermediate_size = self.trg_config.intermediate_size


@dataclass
class IncubationConfig:
    meta_config: Optional[Config]
    trg_config: Optional[Config]
    meta_layer_split: List[int]
    trg_layer_split: List[int]
    special_modules_mapping: Optional[Dict[str, str]] = None
    special_modules_copy_init: bool = False
    use_fused: bool = False
    module_activate_g: Union[int, str] = "None"
    module_activate_f: Union[int, str] = "None"
    replace_layers_index: Optional[int] = None
    
    def __post_init__(self):
        self.block_size = self.meta_config.block_size
        self.vocab_size = self.meta_config.vocab_size
        self.meta_width = self.meta_config.n_embd
        self.meta_depth = self.meta_config.n_layer
        self.meta_intermediate_size = self.meta_config.intermediate_size
        self.trg_width = self.trg_config.n_embd
        self.trg_depth = self.trg_config.n_layer
        self.trg_intermediate_size = self.trg_config.intermediate_size
        self.num_total_trg_layers = len(self.meta_layer_split)


@dataclass
class GradualIncubationConfig(IncubationConfig):
    activate_before_n_layer: Optional[int] = None