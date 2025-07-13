import torch
import torch.nn as nn

def get_model(num_classes: int = 2, pretrained: bool = True):
    print("[i] Loading DETR from Facebook's repo via torch.hub...")
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=pretrained)
    model.class_embed = nn.Linear(model.class_embed.in_features, num_classes)
    return model


def get_device(requested: str or None = None):
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available – falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)
