import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import os
import random

class PennFudanDataset(Dataset):
    def __init__(self, root: str or Path, transforms: bool = True):
        self.root = Path(root)
        self.transforms_flag = transforms
        self.imgs = sorted(os.listdir(self.root / "PNGImages"))
        self.masks = sorted(os.listdir(self.root / "PedMasks"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path   = self.root / "PNGImages" / self.imgs[idx]
        mask_path  = self.root / "PedMasks" / self.masks[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Each pedestrian has unique (R,G,B) value >0, background == 0
        mask = np.array(mask, dtype=np.int32)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        # Split mask per instance
        masks = mask == obj_ids[:, None, None]

        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin, xmax = pos[1].min(), pos[1].max()
            ymin, ymax = pos[0].min(), pos[0].max()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes  = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # pedestrian = 1
        masks  = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        img = F.to_tensor(img)

        if self.transforms_flag:
            if random.random() > 0.5:
                img = F.hflip(img)
                width = img.shape[2]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
