from typing import Annotated

import torch.nn as nn
from base_block import BaseBlock
from mmdet3d.registry import TASK_UTILS
from pydantic import BaseModel, Field
from spikingjelly.activation_based import layer


class BottleNeckConfig(BaseModel):
    # NOTE: seems these parameters cannot be changed to other values
    #       because of expansion hardcoded value
    in_channels: Annotated[int, Field(gt=0)] = 64
    mid_channels: Annotated[int, Field(gt=0)] = 16
    stride: Annotated[int, Field(gt=0)] = 1
    dilation: Annotated[int, Field(gt=0)] = 1
    groups: Annotated[int, Field(gt=0)] = 1
    base_width: Annotated[int, Field(gt=0)] = 64


@TASK_UTILS.register_module()
class BottleNeck(BaseBlock):
    def __init__(self, cfg=None, step_mode="m", device=None):
        self.cfg = BottleNeckConfig(**cfg) if cfg else BottleNeckConfig()
        super().__init__(step_mode=step_mode, device=device)
        self._build()

    def _build(self):
        """
        Architecture
        ------------
        ┌────────────────────┐
        │       Conv         │─┐
        ├────────────────────┤ │
        │ BatchNormalization │ │
        ├────────────────────┤ │
        │    SpikingNeuron   │ │
        ├────────────────────┤ │
        │       Conv         │ │
        ├────────────────────┤ │  # sub-block
        │ BatchNormalization │ │
        ├────────────────────┤ │
        │    SpikingNeuron   │ │
        ├────────────────────┤ │
        │       Conv         │ │
        ├────────────────────┤ │
        │ BatchNormalization │─┘
        └────────────────────┘
        """
        EXPANSION = 4  # final output channels will be 4 times the neck channels width
        width = (  # of the neck of the bottle
            int(self.cfg.mid_channels * (self.cfg.base_width / 64.0)) * self.cfg.groups
        )
        self.bottle_neck = nn.Sequential(
            # ---- Node #1 ------------------------------------------------------------
            layer.Conv2d(  # 1x1 convolution
                in_channels=self.cfg.in_channels,  # 64
                out_channels=width,  # 16 * (64 / 64) * 1 = 16
                kernel_size=1,
                stride=1,
                bias=False,
                step_mode=self.step_mode,
            ),
            layer.BatchNorm2d(
                width,
                step_mode=self.step_mode,
            ),
            self.activation_model(),
            # ---- Node #2 ------------------------------------------------------------
            layer.Conv2d(  # 3x3 convolution
                in_channels=width,  # 16
                out_channels=width,  # 16
                kernel_size=3,
                stride=self.cfg.stride,
                padding=self.cfg.dilation,
                groups=self.cfg.groups,
                bias=False,
                dilation=self.cfg.dilation,
                step_mode=self.step_mode,
            ),
            layer.BatchNorm2d(
                width,
                step_mode=self.step_mode,
            ),
            self.activation_model(),
            # # ---- Node #3 Without Neuron ---------------------------------------------
            layer.Conv2d(  # 1x1 convolution
                in_channels=width,  # 16
                out_channels=self.cfg.mid_channels * EXPANSION,  # 16 * 4 = 64
                kernel_size=1,
                stride=1,
                bias=False,
                step_mode=self.step_mode,
            ),
            layer.BatchNorm2d(
                self.cfg.mid_channels * EXPANSION,
                step_mode=self.step_mode,
            ),
        ).to(device=self.device)

    def forward(self, x):
        out = self.bottle_neck(x)

        return out
