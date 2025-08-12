from typing import Annotated

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from spikingjelly.activation_based import layer
from spikingjelly.clock_driven import neuron

from .blocks import ResBlock


class SpikingResNetConfig(BaseModel):
    arch: Annotated[
        str,
        Field(
            description="ResNet architecture to use",
            pattern="^resnet(18|34|50|101|152)$",
        ),
    ] = "resnet18"

    in_channels: Annotated[int, Field(gt=0)] = 64


class SpikingResNet(nn.Module):
    ARCH = dict(
        resnet18=dict(
            sub_block_type="BasicBlock",
            res_count=[2, 2, 2, 2],  # each ResBlock will be repeated N times
        ),
        resnet34=dict(
            sub_block_type="BasicBlock",
            res_count=[3, 4, 6, 3],
        ),
        resnet50=dict(
            sub_block_type="Bottleneck",
            res_count=[3, 4, 6, 3],
        ),
        resnet101=dict(
            sub_block_type="Bottleneck",
            res_count=[3, 4, 23, 3],
        ),
        resnet152=dict(
            sub_block_type="Bottleneck",
            res_count=[3, 8, 36, 3],
        ),
    )

    def __init__(self, cfg=None, step_mode="m", device=None):
        self.cfg = SpikingResNetConfig(**cfg) if cfg else SpikingResNetConfig()
        super().__init__()
        self._init(step_mode, device)

    def _init(self, step_mode, device):
        # ---- Sanity check -----------------------------------------------------------
        assert step_mode in [
            "m",
            "s",
        ], "step_mode must be 'm' for multi-step or 's' for single-step"
        assert device is None or isinstance(
            device, torch.device
        ), "device must be a torch.device or None"
        # ---- Spiking Neuron ---------------------------------------------------------
        self.step_mode = step_mode
        # ---- Device -----------------------------------------------------------------
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # ---- Build Architecture -----------------------------------------------------
        self._build()

    def _build(self):
        """
        Architecture
        ------------
        """
        arch = self.ARCH[self.cfg.arch]
        spiking_neuron = (
            neuron.MultiStepParametricLIFNode
            if self.step_mode == "m"
            else neuron.ParametricLIFNode
        )

        self.net = nn.Sequential()

        # -----------------------------------------------------------------------------
        # ---- Shared Layers among all ResNets ----------------------------------------
        # -----------------------------------------------------------------------------
        shared = nn.Sequential(
            layer.Conv2d(
                in_channels=self.cfg.in_channels,
                out_channels=self.cfg.in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                step_mode=self.step_mode,
            ),
            layer.BatchNorm2d(
                self.cfg.in_channels,
                step_mode=self.step_mode,
            ),
            spiking_neuron(),
            layer.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                step_mode=self.step_mode,
            ),
        ).to(device=self.device)
        self.net.append(shared)

        # -----------------------------------------------------------------------------
        # ---- ResNet Stages ----------------------------------------------------------
        # -----------------------------------------------------------------------------

        if arch["sub_block_type"] == "BasicBlock":
            # ---- Stage 0 (which is a bit different from next ones) ------------------
            self.net.append(
                self.basic_res_stage(
                    in_channels=self.cfg.in_channels,
                    out_channels=self.cfg.in_channels,
                    count=arch["res_count"][0],
                    half_first_block=False,
                    step_mode=self.step_mode,
                    device=self.device,
                )
            )
            # ---- Subsequent Stages --------------------------------------------------
            later_in_channels = self.cfg.in_channels
            for res_count in arch["res_count"][1:]:
                later_in_channels *= 2  # double the channels for next stage
                self.net.append(
                    self.basic_res_stage(
                        in_channels=later_in_channels,
                        out_channels=later_in_channels,
                        count=res_count,
                        half_first_block=True,  # first block is half size
                        step_mode=self.step_mode,
                        device=self.device,
                    )
                )
        if arch["sub_block_type"] == "Bottleneck":
            # ---- Stage 0 (which is a bit different from next ones) ------------------
            self.net.append(
                self.bottleneck_res_stage(
                    in_channels=self.cfg.in_channels,
                    mid_channels=self.cfg.in_channels,
                    stride=1,
                    count=arch["res_count"][0],
                    double_first_block=False,
                    step_mode=self.step_mode,
                    device=self.device,
                )
            )
            # ---- Subsequent Stages --------------------------------------------------
            later_in_channels = self.cfg.in_channels
            for res_count in arch["res_count"][1:]:
                later_in_channels *= 2
                self.net.append(
                    self.bottleneck_res_stage(
                        in_channels=later_in_channels,
                        mid_channels=later_in_channels,
                        stride=2,
                        count=res_count,
                        double_first_block=True,
                        step_mode=self.step_mode,
                        device=self.device,
                    )
                )

    def extra_repr(self):
        return f"ARCH={self.cfg.arch.upper()}"

    def forward(self, x):
        out = self.net(x)

        return out

    # TODO: Maybe create ResStage class
    @staticmethod
    def basic_res_stage(
        in_channels,
        out_channels,
        count=1,
        half_first_block=False,
        step_mode=None,
        device=None,
    ):
        res_stage = nn.Sequential()

        if half_first_block:
            half_in_channels = in_channels // 2
            res_stage.append(
                ResBlock(
                    cfg=dict(
                        sub_block_cfg=dict(
                            type="BasicBlock",
                            cfg=dict(
                                in_channels=half_in_channels,
                                out_channels=out_channels,
                                stride=2,
                            ),
                        ),
                        skip_block_cfg=dict(
                            type="ConvBlock",
                            cfg=dict(
                                in_channels=half_in_channels,
                                out_channels=out_channels,
                                stride=2,
                            ),
                        ),
                    ),
                    step_mode=step_mode,
                    device=device,
                )
            )
            count -= 1  # reduce count by 1 since we already added one block

        for _ in range(count):
            res_stage.append(
                ResBlock(
                    cfg=dict(
                        sub_block_cfg=dict(
                            type="BasicBlock",
                            cfg=dict(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=1,
                            ),
                        ),
                        skip_block_cfg=dict(type="identity"),
                    ),
                    step_mode=step_mode,
                    device=device,
                )
            )

        return res_stage

    @staticmethod
    def bottleneck_res_stage(
        in_channels,
        mid_channels,
        stride=1,
        dilation=1,
        groups=1,
        count=1,
        double_first_block=False,
        base_width=64,
        step_mode=None,
        device=None,
    ):
        res_stage = nn.Sequential()

        res_stage.append(
            ResBlock(
                cfg=dict(
                    sub_block_cfg=dict(
                        type="BottleNeck",
                        cfg=dict(
                            in_channels=(
                                in_channels
                                if not double_first_block
                                else in_channels * 2
                            ),
                            mid_channels=mid_channels,
                            stride=stride,
                            dilation=dilation,
                            groups=groups,
                            base_width=base_width,
                        ),
                    ),
                    skip_block_cfg=dict(
                        type="ConvBlock",
                        cfg=dict(
                            in_channels=(
                                in_channels
                                if not double_first_block
                                else in_channels * 2
                            ),
                            out_channels=mid_channels * 4,  # expansion factor
                            stride=stride,
                        ),
                    ),
                ),
                step_mode=step_mode,
                device=device,
            )
        )
        count -= 1  # reduce count by 1 since we already added one block

        for _ in range(count):
            res_stage.append(
                ResBlock(
                    cfg=dict(
                        sub_block_cfg=dict(
                            type="BottleNeck",
                            cfg=dict(
                                in_channels=mid_channels * 4,
                                mid_channels=mid_channels,
                                stride=1,
                                dilation=dilation,
                                groups=groups,
                                base_width=base_width,
                            ),
                        ),
                        skip_block_cfg=dict(type="identity"),
                    ),
                    step_mode=step_mode,
                    device=device,
                )
            )

        return res_stage
