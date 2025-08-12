from spikingjelly.clock_driven import neuron
from torch import nn
import torch

from mmdet3d.registry import MODELS
from .spiking_resnet import SpikingResNet


@MODELS.register_module()
class SpikingResNetFeatures(SpikingResNet):
    selected_layers = [
        "net.0.2: MultiStepParametricLIFNode",
        "net.1.1.sub_block.basic_block.2: MultiStepParametricLIFNode",
        "net.2.1.sub_block.basic_block.2: MultiStepParametricLIFNode",
        "net.3.1.sub_block.basic_block.2: MultiStepParametricLIFNode",
        "net.4.1.sub_block.basic_block.2: MultiStepParametricLIFNode",
    ]

    def __init__(self, cfg=None, step_mode="m", device=None):
        super().__init__(cfg=cfg, step_mode=step_mode, device=device)
        # ---- Feature Maps -----------------------------------------------------------
        self.feature_maps = dict()
        add_feature_maps_hook(self, self.selected_layers)

    def forward(self, x):
        # self.feature_maps.clear()
        x = x.to(torch.float32)
        _ = self.net(x)  # Trigger the forward pass to collect feature maps
        list_fmaps = list(self.feature_maps.values())

        return list_fmaps  # List of feature maps from selected layers


def add_feature_maps_hook(model, selected_layers):
    def create_hook(layer_name):
        def hook(module, input, output):
            # it is important to detach the output
            # to avoid computation graph issues (memory leaks)
            model.feature_maps[layer_name] = output.detach()

        return hook

    for layer_name in selected_layers:
        layer_name = layer_name.split(":")[0]
        layer = model.get_submodule(layer_name)
        layer.register_forward_hook(create_hook(layer_name))


def print_selected_layer_names(model):
    """
    Just define the target class type
    you will get <layer_path>: <layer_class_name>
    copy-paste the output to `selected_layers` list
    """
    TARGET_CLASS_TYPE = neuron.MultiStepParametricLIFNode

    for name, layer in model.named_modules():
        if isinstance(layer, TARGET_CLASS_TYPE):
            print(f"{name}: {layer.__class__.__name__}")
