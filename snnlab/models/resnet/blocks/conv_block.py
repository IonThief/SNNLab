from typing import Annotated

import torch
import torch.nn as nn
from mmdet3d.registry import TASK_UTILS
from pydantic import BaseModel, Field
from spikingjelly.activation_based import layer


class ConvBlockConfig(BaseModel):
    in_channels: Annotated[int, Field(gt=0)] = 64
    out_channels: Annotated[int, Field(gt=0)] = 64
    stride: Annotated[int, Field(gt=0)] = 1


@TASK_UTILS.register_module()
class ConvBlock(nn.Module):
    def __init__(self, cfg=None, step_mode=None, device=None):
        self.cfg = ConvBlockConfig(**cfg) if cfg else ConvBlockConfig()
        super().__init__()
        self._init(cfg, step_mode, device)
        self._build()

    def _init(self, cfg, step_mode, device):
        # ---- Sanity check -----------------------------------------------------------
        # TODO:
        # ---- Spiking Neuron ---------------------------------------------------------
        self.step_mode = step_mode
        # ---- Device -----------------------------------------------------------------
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _build(self):
        """
        Architecture
        ------------
        ┌────────────────────┐
        │      1x1Conv       │─┐
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
