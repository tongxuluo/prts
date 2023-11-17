import glob
import os
from pathlib import Path
import torch
import torch.nn as nn

def incremental_load(trg_model: nn.Module, trg_init_path: Path):
    if isinstance(trg_init_path, str):
        trg_init_path = Path(trg_init_path)
    pattern = os.path.join(trg_init_path, "iter-*.pth")
    ckpts = glob.glob(pattern)
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        state_dict = torch.load(latest_ckpt)
        model_state_dict = state_dict["model"]
        to_load_dict = {k: model_state_dict[k] for k in model_state_dict if "trg_layers" in k}
        trg_model.load_state_dict(to_load_dict, strict=False)