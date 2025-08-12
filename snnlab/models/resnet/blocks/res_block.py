from typing import Annotated

import torch
from pydantic import BaseModel, Field
from spikingjelly.clock_driven import neuron
from torch import nn

from .block_builder import BlockBuilder


class ResBlockConfig(BaseModel):
    sub_block_cfg: Annotated[
        dict,
        Field(description="it should follow MMLab Convention"),
    ] = dict(
        type="BasicBlock",
        cfg=dict(
            in_channels=64,
            out_channels=64,
            stride=1,
        ),
    )
    skip_block_cfg: Annotated[
        dict,
        Field(description="it should follow MMLab Convention"),
    ] = dict(
        type="identity",
    )


class ResBlock(nn.Module):
    """
    Architecture
    ------------
      ┌───┐
      │ x │
      └─┬─┘
        ├───────────┐
        │           │
    ┌───┴────┐  ┌───┴─────┐
    │SubBlock│  │SkipBlock│ # downsampled input, if given skip_block
    └───┬────┘  └───┬─────┘ # skip_block can be either:
        │           │       #   - identity (Nothing)
        ▼           │       #   - convolutional (1x1 Conv, BatchNorm)
      ┌───┐         │
      │Add├─────────┘
      └─┬─┘
        │
        ▼
    ┌──────────┐
    │Activation│
    └───┬──────┘
        │
        ▼
     ┌─────┐
     │ Out │
     └─────┘
    """

    def __init__(self, cfg=None, step_mode=None, device=None):
        self.cfg = ResBlockConfig(**cfg) if cfg else ResBlockConfig()
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
        builder = BlockBuilder(
            cfg=dict(
                activation_model=self.activation_model,
            ),
            step_mode=self.step_mode,
            device=self.device,
        )
        self.activation_neuron = self.activation_model().to(self.device)
        self.sub_block = builder.build(self.cfg.sub_block_cfg)
        if self.cfg.skip_block_cfg["type"] == "identity":
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = builder.build(self.cfg.skip_block_cfg)

    def forward(self, x):
        skip_block_out = self.skip_layer(x)
        sub_block_out = self.sub_block(x)
        out = sub_block_out + skip_block_out
        out = self.activation_neuron(out)

        return out

    @property
    def activation_model(self):
        spiking_neuron_model = (
            neuron.MultiStepParametricLIFNode
            if self.step_mode == "m"
            else neuron.ParametricLIFNode
        )
        return spiking_neuron_model
