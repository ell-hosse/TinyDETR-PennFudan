import torch.nn as nn
from transformers import DetrImageProcessor, DetrForObjectDetection

class DETRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # switch to new ImageProcessor
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1, # your number of classes
            ignore_mismatched_sizes=True
        )

    def forward(self, images, targets=None):
        """
        images: list of PIL.Image or tensors [3,H,W]
        targets: list of dicts with 'class_labels' and 'boxes' (normalized)
        """
        inputs = self.processor(images=images, return_tensors="pt").to(images[0].device)

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
