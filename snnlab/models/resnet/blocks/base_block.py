from abc import abstractmethod

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron


class BaseBlock(nn.Module):
    """
    Base class for all blocks in the Spiking ResNet architecture.
    This class defines the basic structure and methods that all blocks should implement.
    """

    def __init__(self, step_mode=None, device=None):
        super().__init__()
        self.step_mode = step_mode
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(self, x):
        pass

    @abstractmethod
    def _build(self):
        """
        Build the block architecture.
        This method should be implemented by subclasses
        to define the specific architecture of the block.
        """
        raise NotImplementedError("Build method must be implemented by subclasses.")

    @property
    def activation_model(self):
        # TODO: Change logic if you want to support other neuron models
        #       maybe add a config option
        _activation_model = (
            neuron.MultiStepParametricLIFNode
            if self.step_mode == "m"
            else neuron.ParametricLIFNode
        )
        return _activation_model
