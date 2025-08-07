from typing import Annotated

import torch.nn as nn
from base_block import BaseBlock
from mmdet3d.registry import TASK_UTILS
from pydantic import BaseModel, Field
from spikingjelly.activation_based import layer


class ConvBlockConfig(BaseModel):
    in_channels: Annotated[int, Field(gt=0)] = 64
    out_channels: Annotated[int, Field(gt=0)] = 64
    stride: Annotated[int, Field(gt=0)] = 1


@TASK_UTILS.register_module()
class ConvBlock(BaseBlock):
    def __init__(self, cfg=None, step_mode=None, device=None):
        self.cfg = ConvBlockConfig(**cfg) if cfg else ConvBlockConfig()
        super().__init__(step_mode=step_mode, device=device)
        self._build()

    def _build(self):
        """
        Architecture
        ------------
        ┌────────────────────┐
        │        Conv        │─┐
        ├────────────────────┤ │ # downsampling layer
        │ BatchNormalization │─┘
        └────────────────────┘
        """
        self.conv_block = nn.Sequential(
            layer.Conv2d(  # 1x1 convolution
                in_channels=self.cfg.in_channels,
                out_channels=self.cfg.out_channels,
                kernel_size=1,
                stride=self.cfg.stride,
                bias=False,
                step_mode=self.step_mode,
            ),
            layer.BatchNorm2d(
                self.cfg.out_channels,
                step_mode=self.step_mode,
            ),
        ).to(device=self.device)

    def forward(self, x):
        out = self.conv_block(x)

        return out
