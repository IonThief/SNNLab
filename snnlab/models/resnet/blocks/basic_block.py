from typing import Annotated

import torch.nn as nn
from base_block import BaseBlock
from mmdet3d.registry import TASK_UTILS
from pydantic import BaseModel, Field
from spikingjelly.activation_based import layer


class BasicBlockConfig(BaseModel):
    in_channels: Annotated[int, Field(gt=0)] = 64
    out_channels: Annotated[int, Field(gt=0)] = 64
    stride: Annotated[int, Field(gt=0)] = 1


@TASK_UTILS.register_module()
class BasicBlock(BaseBlock):
    def __init__(self, cfg=None, step_mode=None, device=None):
        self.cfg = BasicBlockConfig(**cfg) if cfg else BasicBlockConfig()
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
        │    SpikingNeuron   │ │  # sub-block
        ├────────────────────┤ │
        │       Conv         │ │
        ├────────────────────┤ │
        │ BatchNormalization │─┘
        └────────────────────┘
        """
        # ---- Constants --------------------------------------------------------------
        GROUPS = 1  # Must be 1
        DILATION = 1  # Must be 1
        # ---- Sub-block --------------------------------------------------------------
        self.basic_block = nn.Sequential(
            layer.Conv2d(
                in_channels=self.cfg.in_channels,
                out_channels=self.cfg.out_channels,
                kernel_size=3,
                stride=self.cfg.stride,
                padding=DILATION,
                groups=GROUPS,
                bias=False,
                dilation=DILATION,
                step_mode=self.step_mode,
            ),
            layer.BatchNorm2d(self.cfg.out_channels, step_mode=self.step_mode),
            self.activation_model(),
            layer.Conv2d(
                in_channels=self.cfg.out_channels,
                out_channels=self.cfg.out_channels,
                kernel_size=3,
                stride=1,
                padding=DILATION,
                groups=GROUPS,
                bias=False,
                dilation=DILATION,
                step_mode=self.step_mode,
            ),
            layer.BatchNorm2d(self.cfg.out_channels, step_mode=self.step_mode),
        ).to(device=self.device)

    def forward(self, x):
        out = self.basic_block(x)

        return out
