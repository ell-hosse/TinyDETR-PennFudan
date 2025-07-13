import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

def get_model(num_classes=2, pretrained=True):
    if pretrained:
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=num_classes)
    else:
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)
        model.class_labels_classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    return model

def get_processor():
    return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def get_device(requested: str or None = None):
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available – falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)
