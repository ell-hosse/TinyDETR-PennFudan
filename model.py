import torch.nn as nn
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

class DETRModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50",
                                                            do_rescale=True,
                                                            do_resize = False
                                                            )
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,
            ignore_mismatched_sizes=True
        ).to(self.device)

    def forward(self, images, targets=None):
        """
        images: list of PIL.Image or tensors [3,H,W]
        targets: list of dicts with 'class_labels' and 'boxes' (normalized)
        """
        inputs = self.processor(
            images=images,
            annotations=targets,
            return_tensors="pt"
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.device)

        if targets is not None:
            # HuggingFace expects 'labels' key
            inputs["labels"] = [
                {"class_labels": t["class_labels"], "boxes": t["boxes"]}
                for t in targets
            ]

        outputs = self.model(**inputs)
        return outputs

    def compute_loss(self, outputs):
        return outputs.loss # DETR’s built-in loss
