import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import DATA_ROOT, BATCH_SIZE
from transformers import DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class PennFudanDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image & mask
        img_path  = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Extract instances
        obj_ids = np.unique(mask)[1:]  # drop background=0
        masks = mask == obj_ids[:, None, None]  # shape [N, H, W]

        boxes = []
        for m in masks:
            pos = np.where(m)
            x0, y0 = pos[1].min(), pos[0].min()
            x1, y1 = pos[1].max(), pos[0].max()
            boxes.append([x0, y0, x1, y1])

        area = [(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in boxes]
        labels = [1] * len(boxes)
        iscrowd = [0] * len(boxes)

        target = {
            "image_id": idx,
            "annotations": [
                {
                    "bbox": box,
                    "category_id": label,
                    "area": a,
                    "iscrowd": c,
                }
                for box, label, a, c in zip(boxes, labels, area, iscrowd)
            ]
        }

        # DETR Hugging Face expects PIL input, returns pixel values + encoding
        processed = processor(images=img, annotations=target, return_tensors="pt")
        processed["pixel_values"] = processed["pixel_values"].squeeze(0)  # [1,3,H,W] → [3,H,W]
        return processed

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}

def get_dataloaders():
    dataset = PennFudanDataset(DATA_ROOT)
    n = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_dl, val_dl
