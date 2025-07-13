# model.py
import torch
import torch.nn as nn
from transformers import DetrForObjectDetection

def get_model(num_classes=2, pretrained=True):
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    hid = model.class_labels_classifier.in_features
    model.class_labels_classifier = nn.Linear(hid, num_classes)
    model.bbox_predictor = nn.Linear(hid, 4)

    return model

def get_device(requested: str or None = None):
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA requested but not available – falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)
