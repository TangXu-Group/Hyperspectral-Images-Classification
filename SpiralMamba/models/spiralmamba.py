import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_vssm_model


class SpiralMamba(nn.Module):
    def __init__(self, config):
        super(SpiralMamba, self).__init__()
        self.spiralmamba = build_vssm_model(config)

    def forward(self, x):
        outputs = self.spiralmamba(x)
        return outputs