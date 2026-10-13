from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin

from models.motion_module import zero_module
from models.resnet import InflatedConv3d


class ConvIn(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        # block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, conditioning_embedding_channels, kernel_size=3, padding=1
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        return embedding
