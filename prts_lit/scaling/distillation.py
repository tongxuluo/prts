import torch
import torch.nn as nn
from ..utils import (
    DistillationConfig,
)
from lit_gpt.model import GPT

class DistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model, config: DistillationConfig) -> None:
        super().__init__()
        self.config = config
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.post_init()
    
    def post_init(self):
        for n, p in self.teacher_model.named_parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        src_model: GPT,
        trg_model: GPT,
        config: DistillationConfig,
    ):
        with torch.no_grad():
            distill_model = cls(src_model, trg_model, config)
        return distill_model

    def forward(self, *args, **kwargs):
        teacher_output = self.teacher_model(*args, **kwargs)
        student_output = self.student_model(*args, **kwargs)
        return teacher_output, student_output