from dataclasses import dataclass, asdict, field
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from lit_gpt.model import Config


@dataclass
class PrtsConfig:
    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_path):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        output_dict = self.__dict__
        output_path = save_path

        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, config_path):

        if os.path.isfile(config_path):
            config_file = config_path
        else:
            raise ValueError(f"Can't find '{config_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        config = cls()

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.post_init()
        return config

    @classmethod
    def from_json_file(cls, path_json_file):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)
        return json_object

@dataclass
class bert2BERTConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})
    from_left: bool = field(default=False, metadata={"help": "select from left"})
    inner_stacking: bool = field(default=False, metadata={"help": "112233"})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.src_width = src_config.n_embd
        self.src_depth = src_config.n_layer
        self.src_head = src_config.n_head
        self.src_query_groups = src_config.n_query_groups
        self.src_intermediate_size = src_config.intermediate_size
        self.trg_width = trg_config.n_embd
        self.trg_depth = trg_config.n_layer
        self.trg_head = trg_config.n_head
        self.trg_query_groups = trg_config.n_query_groups
        self.trg_intermediate_size = trg_config.intermediate_size


@dataclass
class IncubationConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the meta model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})
    meta_layer_split: List[int] = field(default=None, metadata={"help": "The split of the meta model."})
    trg_layer_split: List[int] = field(default=None, metadata={"help": "The split of the target model."})
    special_modules_mapping: Optional[Dict[str, str]] = field(default=None, metadata={"help": "The mapping of special module."})
    special_modules_copy_init: bool = field(default=False, metadata={"help": "Copy init the special module."})
    module_activate_g: Union[int, str] = field(default="None", metadata={"help": "The layer add g operation."})
    module_activate_f: Union[int, str] = field(default="None", metadata={"help": "The layer add f operation."})
    replace_layers_index: Optional[int] = field(default=None, metadata={"help": "The layer index to replace."})
    
    def post_init(self):
        meta_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = meta_config.block_size
        self.vocab_size = meta_config.vocab_size
        self.meta_width = meta_config.n_embd
        self.meta_depth = meta_config.n_layer
        self.meta_intermediate_size = meta_config.intermediate_size
        self.trg_width = trg_config.n_embd
        self.trg_depth = trg_config.n_layer
        self.trg_intermediate_size = trg_config.intermediate_size
        self.num_total_trg_layers = len(self.meta_layer_split)


@dataclass
class GradualIncubationConfig(IncubationConfig):
    activate_before_n_layer: Optional[int] = field(default=None, metadata={"help": "Activate before n layers."})


@dataclass
class LiGOConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.src_width = src_config.n_embd
        self.src_depth = src_config.n_layer
        self.src_head = src_config.n_head
        self.src_query_groups = src_config.n_query_groups
        self.src_intermediate_size = src_config.intermediate_size
        self.trg_width = trg_config.n_embd
        self.trg_depth = trg_config.n_layer
        self.trg_head = trg_config.n_head
        self.trg_query_groups = trg_config.n_query_groups
        self.trg_intermediate_size = trg_config.intermediate_size


@dataclass
class SolarConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})
    num_drop_layers: int = field(default=None, metadata={"help": "The number of layers to drop."})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        assert src_config.n_embd == trg_config.n_embd
        self.src_depth = src_config.n_layer
        self.trg_depth = trg_config.n_layer


@dataclass
class StackingptConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})
    src_init_path: str = field(default=None, metadata={"help": "The list of the source model path."})
    embd_name: str = field(default=None, metadata={"help": "The Model's embedding name"})
    ln_name: str = field(default=None, metadata={"help": "The Model's ln name"})
    head_name: str = field(default=None, metadata={"help": "The Model's head name"})
    layer_name: str = field(default=None, metadata={"help": "The Model's layer name"})
    stacking_list: Optional[List] = field(default=None, metadata={"help": "layer idx"})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.src_width = src_config.n_embd
        self.src_depth = src_config.n_layer
        self.src_head = src_config.n_head
        self.src_query_groups = src_config.n_query_groups
        self.src_intermediate_size = src_config.intermediate_size
        self.trg_width = trg_config.n_embd
        self.trg_depth = trg_config.n_layer
        self.trg_head = trg_config.n_head
        self.trg_query_groups = trg_config.n_query_groups
        self.trg_intermediate_size = trg_config.intermediate_size

@dataclass
class MilkConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the meta model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})
    meta_layer_split: List[int] = field(default=None, metadata={"help": "The split of the meta model."})
    trg_layer_split: List[int] = field(default=None, metadata={"help": "The split of the target model."})
    special_modules_mapping: Optional[Dict[str, str]] = field(default=None, metadata={"help": "The mapping of special module."})

    def post_init(self):
        meta_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = meta_config.block_size
        self.vocab_size = meta_config.vocab_size
        self.meta_width = meta_config.n_embd
        self.meta_depth = meta_config.n_layer
        self.meta_intermediate_size = meta_config.intermediate_size
        self.trg_width = trg_config.n_embd
        self.trg_depth = trg_config.n_layer
        self.trg_intermediate_size = trg_config.intermediate_size
        self.num_total_trg_layers = len(self.meta_layer_split)


@dataclass
class GradualStakcingConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    gs_layers_num: List[int] = field(default=None, metadata={"help": "The layers to grow."})
    grow_step_interval: int = field(default=5000, metadata={"help": "grow step interval."})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.src_width = src_config.n_embd
        self.src_depth = src_config.n_layer
        self.src_head = src_config.n_head
        self.src_query_groups = src_config.n_query_groups
        self.src_intermediate_size = src_config.intermediate_size

@dataclass
class DistillationConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.meta_width = src_config.n_embd
        self.meta_depth = src_config.n_layer
        self.meta_intermediate_size = src_config.intermediate_size
        self.trg_width = trg_config.n_embd
        self.trg_depth = trg_config.n_layer
        self.trg_intermediate_size = trg_config.intermediate_size

@dataclass
class MsgConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})
    src_path: str = field(default=None, metadata={"help": "The list of the source model path."})
    grow_step: int = field(default=5000, metadata={"help": "grow step interval."})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        trg_config = Config.from_name(self.trg_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.src_width = src_config.n_embd
        self.src_depth = src_config.n_layer
        self.src_head = src_config.n_head
        self.src_query_groups = src_config.n_query_groups
        self.src_intermediate_size = src_config.intermediate_size
        self.trg_width = trg_config.n_embd
        self.trg_depth = trg_config.n_layer
        self.trg_head = trg_config.n_head
        self.trg_query_groups = trg_config.n_query_groups
        self.trg_intermediate_size = trg_config.intermediate_size


@dataclass
class MsltConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    mslt_layers_num: List[int] = field(default=None, metadata={"help": "The layers to grow."})
    grow_step_interval: int = field(default=5000, metadata={"help": "grow step interval."})

    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.src_width = src_config.n_embd
        self.src_depth = src_config.n_layer
        self.src_head = src_config.n_head
        self.src_query_groups = src_config.n_query_groups
        self.src_intermediate_size = src_config.intermediate_size

@dataclass
class ZeroConfig(PrtsConfig):
    src_config_name: str = field(default=None, metadata={"help": "The name of the source model config to use."})
    trg_config_name: str = field(default=None, metadata={"help": "The name of the target model config to use."})
    src_path: str = field(default=None, metadata={"help": "The list of the source model path."})
    
    def post_init(self):
        src_config = Config.from_name(self.src_config_name)
        self.block_size = src_config.block_size
        self.vocab_size = src_config.vocab_size
        self.src_width = src_config.n_embd
        self.src_depth = src_config.n_layer
        self.src_head = src_config.n_head
        self.src_query_groups = src_config.n_query_groups
        self.src_intermediate_size = src_config.intermediate_size