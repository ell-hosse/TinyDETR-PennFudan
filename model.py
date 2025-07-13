import torch
import torchvision
from torchvision.models.detection import detr_resnet50
from torchvision.models.detection.transform import GeneralizedRCNNTransform

def get_model(num_classes: int, pretrained: bool = True):
    model = detr_resnet50(pretrained=pretrained, num_classes=num_classes)
    return model

def get_device(requested: str or None = None):
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available – falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)
