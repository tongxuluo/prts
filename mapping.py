from .utils import (
    bert2BERTConfig,
    IncubationConfig,
    GradualIncubationConfig,
)
from scaling import (
    bert2bert,
    Incubation,
    GradualIncubation,
)

PRTS_CONFIG_TO_SCALING_MAPPING = {
    bert2BERTConfig: bert2bert,
    IncubationConfig: Incubation,
    GradualIncubationConfig: GradualIncubation,
}


def get_prts_model(src_or_meta_model, trg_model=None, prts_config=None, *args, **kwargs):
    if type(prts_config) in PRTS_CONFIG_TO_SCALING_MAPPING.keys():
        if trg_model is None:
            return PRTS_CONFIG_TO_SCALING_MAPPING[type(prts_config)].from_pretrained(src_or_meta_model, prts_config, *args, **kwargs)
        else:
            return PRTS_CONFIG_TO_SCALING_MAPPING[type(prts_config)](src_or_meta_model, trg_model, prts_config)
