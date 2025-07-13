import torch
import torch.nn as nn

def get_model(num_classes: int, pretrained: bool = True):
    try:
        from torchvision.models.detection import detr_resnet50
        return detr_resnet50(weights=None, num_classes=num_classes)
    except:
        print("[!] DETR not available in torchvision ≥0.20 -> using torch.hub fallback")
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
