# visualize.py
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, utils
from model import DETRModel
from dataset import PennFudanDataset, collate_fn
from config import DATA_ROOT, DEVICE

model = DETRModel().to(DEVICE)
ckpt   = torch.load("detr_pennfudan.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((800,800)),
    transforms.ToTensor(),
])
full_ds = PennFudanDataset(DATA_ROOT, transforms=transform)
n = len(full_ds)
split = int(0.8*n)
val_ds = torch.utils.data.Subset(full_ds, list(range(split, n)))

indices = random.sample(range(len(val_ds)), k=min(10, len(val_ds)))
samples = [val_ds[i] for i in indices]

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4*len(samples)))
for row, (img, target) in enumerate(samples):
    img_cpu = img.cpu()
    axes[row,0].imshow(img_cpu.permute(1,2,0))
    axes[row,0].set_title("Original Image")
    axes[row,0].axis("off")

    # reconstruct mask from target["boxes"] & target from PennFudanDataset
    mask = Image.open(
        f"{DATA_ROOT}/PedMasks/{full_ds.masks[split+indices[row]]}"
    )
    axes[row,1].imshow(mask, cmap="gray")
    axes[row,1].set_title("GT Segmentation Mask")
    axes[row,1].axis("off")

    # DETR expects list of images
    with torch.no_grad():
        outputs = model([img.to(DEVICE)], targets=None)
    # logits [1,Q,C+1], boxes [1,Q,4]
    logits = outputs.logits.softmax(-1)[0,...-1]  # no-object scores at last class
    scores, labels = outputs.logits[0].softmax(-1).max(-1)
    boxes = outputs.pred_boxes[0]  # normalized xyxy

    # select high-confidence detections
    keep = scores > 0.5
    boxes = boxes[keep]
    scores= scores[keep]

    # denormalize boxes to image size
    H,W = img.shape[1], img.shape[2]
    boxes = boxes.cpu().numpy()
    boxes[:,[0,2]] *= W
    boxes[:,[1,3]] *= H

    # overlay boxes
    axes[row,2].imshow(img_cpu.permute(1,2,0))
    for box, sc in zip(boxes, scores):
        x1,y1,x2,y2 = box
        rect = plt.Rectangle(
            (x1,y1), x2-x1, y2-y1,
            fill=False, edgecolor="r", linewidth=2
        )
        axes[row,2].add_patch(rect)
        axes[row,2].text(
            x1, y1-5, f"{sc:.2f}",
            color="yellow", backgroundcolor="black", fontsize=8
        )
    axes[row,2].set_title("Predicted Boxes")
    axes[row,2].axis("off")

plt.tight_layout()
plt.show()
