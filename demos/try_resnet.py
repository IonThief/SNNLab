import torch

from snnlab.models.resnet import SpikingResNet, SpikingResNetFeatures
from snnlab.utils import draw_model_graph


def try_resnet():
    model = SpikingResNet(
        cfg=dict(arch="resnet18", in_channels=3),
        step_mode="m",
    )
    input_size = (1, 1, 3, 224, 224)
    draw_model_graph(
        model,
        input_size=input_size,
        graph_depth=2,
        fname=model.cfg.arch + "_" + model.step_mode + "_step",
    )


def try_resnet_features():
    # WARN:
    # draw_graph function does not work with SpikingResNetFeatures
    # most probably due to the hooks used to collect feature maps

    model = SpikingResNetFeatures(
        cfg=dict(arch="resnet18", in_channels=3),
        step_mode="m",
    )
    input_size = (1, 1, 3, 224, 224)
    x = torch.randn(input_size).to(model.device)
    feature_maps = model(x)
    print(f"Feature maps collected from {len(feature_maps)} layers.")


if __name__ == "__main__":
    try_resnet()
    try_resnet_features()
