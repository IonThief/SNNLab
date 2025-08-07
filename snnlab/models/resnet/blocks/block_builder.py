import torch
from basic_block import BasicBlock  # noqa
from bottle_neck import BottleNeck  # noqa
from conv_block import ConvBlock  # noqa
from mmdet3d.registry import TASK_UTILS
from pydantic import BaseModel, model_validator
from spikingjelly.clock_driven import neuron


class BlockBuilderConfig(BaseModel):
    # TODO: NOT USED YET
    activation_model: type = neuron.MultiStepParametricLIFNode

    # ---- Post-Validation ------------------------------------------------------------
    @model_validator(mode="after")
    def sanity_check(self):
        neuron_name = self.activation_model.__name__
        if not hasattr(neuron, neuron_name):
            raise ValueError(
                f"activation_model ({neuron_name}) must be"
                f" one of the neuron attributes in spikingjelly.clock_driven.neuron"
            )
        return self


class BlockBuilder:
    def __init__(self, cfg=None, step_mode="m", device=None):
        self.cfg = BlockBuilderConfig(**cfg) if cfg else BlockBuilderConfig()
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

    def build(self, cfg):
        cfg["device"] = self.device
        cfg["step_mode"] = self.step_mode
        block = TASK_UTILS.build(cfg)

        return block
