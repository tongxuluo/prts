from .utils import (
    bert2BERTConfig,
    IncubationConfig,
    GradualIncubationConfig,
    LiGOConfig,
    SolarConfig,
    StackingptConfig,
    MilkConfig,
    GradualStakcingConfig,
    DistillationConfig,
    MsgConfig,
    MsltConfig,
    ZeroConfig,
)
from .scaling import (
    bert2BERT,
    IncubationModel,
    GradualIncubation,
    LiGOModel,
    SolarModel,
    Stackingpt,
    MilkModel,
    GradualStackingModel,
    DistillationModel,
    MsgModel,
    MsltModel,
    ZeroModel
)

PRTS_CONFIG_TO_SCALING_MAPPING = {
    bert2BERTConfig: bert2BERT,
    IncubationConfig: IncubationModel,
    GradualIncubationConfig: GradualIncubation,
    LiGOConfig: LiGOModel,
    SolarConfig: SolarModel,
    StackingptConfig: Stackingpt,
    MilkConfig: MilkModel,
    GradualStakcingConfig: GradualStackingModel,
    DistillationConfig: DistillationModel,
    MsgConfig: MsgModel,
    MsltConfig: MsltModel,
    ZeroConfig: ZeroModel,
}


def get_prts_model(src_or_meta_model, trg_model=None, prts_config=None, *args, **kwargs):
    if type(prts_config) in PRTS_CONFIG_TO_SCALING_MAPPING.keys():
        if src_or_meta_model is None and trg_model is None:
            return PRTS_CONFIG_TO_SCALING_MAPPING[type(prts_config)].from_pretrained(prts_config, *args, **kwargs)
        elif trg_model is None:
            return PRTS_CONFIG_TO_SCALING_MAPPING[type(prts_config)].from_pretrained(src_or_meta_model, prts_config, *args, **kwargs)
        elif src_or_meta_model is None:
            return PRTS_CONFIG_TO_SCALING_MAPPING[type(prts_config)].from_pretrained(trg_model, prts_config)
        else:
            return PRTS_CONFIG_TO_SCALING_MAPPING[type(prts_config)](src_or_meta_model, trg_model, prts_config)
