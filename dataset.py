import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import DATA_ROOT, BATCH_SIZE
from transformers import DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class PennFudanDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.imgs  = sorted(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks",  self.masks[idx])
        img  = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        obj_ids = np.unique(mask)[1:]   # drop background=0
        boxes   = []
        labels  = []
        areas   = []
        iscrowd = []

        for oid in obj_ids:
            m = mask == oid
            ys, xs = np.where(m)
            x0, y0 = xs.min(), ys.min()
            x1, y1 = xs.max(), ys.max()
            boxes.append([x0, y0, x1, y1])
            labels.append(1)              # single class: pedestrian
            areas.append((x1 - x0) * (y1 - y0))
            iscrowd.append(0)

        target = {"image_id": idx,
                  "annotations": [
                    {"bbox": b, "category_id": lab,
                     "area": a, "iscrowd": c}
                    for b, lab, a, c in zip(boxes, labels, areas, iscrowd)
                  ]}

        encoding = processor(images=img,
                             annotations=target,
                             return_tensors="pt")
        encoding = {
            "pixel_values": encoding["pixel_values"].squeeze(0),  # → [3, H, W]
            "pixel_mask":   encoding["pixel_mask"].squeeze(0),    # → [H, W]
            "labels":       encoding["labels"][0]                 # dict with tensors
        }
        return encoding

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    pixel_masks = [item["pixel_mask"]   for item in batch]
    labels = [item["labels"]       for item in batch]

    batch_encoding = processor.pad(
        {
            "pixel_values": pixel_values,
            "pixel_mask":   pixel_masks,
            "labels":       labels
        },
        return_tensors="pt"
    )
    return batch_encoding

def get_dataloaders():
    ds = PennFudanDataset(DATA_ROOT)
    n  = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n, len(ds) - n])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)
    return train_dl, val_dl
