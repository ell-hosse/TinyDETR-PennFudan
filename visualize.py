import argparse, random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

import config as C
from dataset import PennFudanDataset
from model import get_model, get_device

PALETTE = ["red", "lime", "yellow", "cyan", "magenta"]

def plot_prediction(img, boxes, scores):
    fig, ax = plt.subplots(figsize=(6,9))
    ax.imshow(img.permute(1,2,0))
    for i, (box,s) in enumerate(zip(boxes, scores)):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=PALETTE[i % len(PALETTE)],
            facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 3, f"{s:.2f}", color=rect.get_edgecolor(),
                fontsize=9, weight="bold")
    ax.axis("off")
    plt.show()

def main(args):
    device = get_device(args.device)
    dataset = PennFudanDataset(C.DATA_ROOT, transforms=False)

    model = get_model(num_classes=C.NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()

    # pick random images
    indices = random.sample(range(len(dataset)), k=args.num_images)
    for idx in indices:
        img, _ = dataset[idx]
        with torch.no_grad():
            out = model([img.to(device)])[0]

        keep = out["scores"] > C.CONF_THRESHOLD
        boxes = out["boxes"][keep].cpu()
        scores = out["scores"][keep].cpu()
        plot_prediction(img, boxes, scores)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/best.pth",
                    help="Path to checkpoint .pth")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num_images", type=int, default=3)
    args = ap.parse_args()
    main(args)
