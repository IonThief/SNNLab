from typing import Annotated

import torch
from block_builder import BlockBuilder
from pydantic import BaseModel, Field
from spikingjelly.clock_driven import neuron
from torch import nn


class ResBlockConfig(BaseModel):
    sub_block_cfg: Annotated[
        dict,
        Field(description="It should follow MMLab Convention"),
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
        Field(description="It should follow MMLab Convention"),
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
                activation_model=self.activation_model,  # WARN: NOT USED though
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


if __name__ == "__main__":
    import os

    OUTPUT_DIR = ".RENDERED_MODELS_AS_GRAPH"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def export_model_architecture_svg(model, input_size):
        from torchview import draw_graph

        draw_graph(
            model,
            input_size=input_size,
            depth=1,
            expand_nested=True,
            graph_dir="TB",
        ).visual_graph.render(
            filename=f"{model.__class__.__name__}",
            directory=OUTPUT_DIR,
            format="svg",
            cleanup=True,
        )
        print(f":: Model {model.__class__.__name__} graph is saved at {OUTPUT_DIR}")

    # Example usage
    # model = ResBlock(
    #     cfg=dict(
    #         sub_block_cfg=dict(
    #             type="BasicBlock",
    #             cfg=dict(
    #                 in_channels=64,
    #                 out_channels=64,
    #                 stride=1,
    #             ),
    #         ),
    #         skip_block_cfg=dict(
    #             type="ConvBlock",
    #             cfg=dict(
    #                 in_channels=64,
    #                 out_channels=64,
    #             ),
    #         ),
    #     ),
    #     step_mode="m",
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # )

    model = ResBlock(
        cfg=dict(
            sub_block_cfg=dict(
                type="BottleNeck",
                cfg=dict(
                    in_channels=64,
                    mid_channels=16,
                    stride=1,
                    dilation=1,
                    groups=1,
                    base_width=64,
                ),
            ),
            # skip_block_cfg=dict(
            #     type="ConvBlock",
            #     cfg=dict(
            #         in_channels=64,
            #         out_channels=64,
            #     ),
            # ),
        ),
        step_mode="m",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    print(model)
    input_size = (
        1,
        1,
        64,
        224,
        224,
    )  # Example input size: (batch_size, channels, height, width)
    export_model_architecture_svg(model, input_size)
