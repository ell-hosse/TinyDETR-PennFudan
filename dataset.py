import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import DATA_ROOT, BATCH_SIZE

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
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

        # Each distinct ID >0 is one pedestrian
        obj_ids = np.unique(mask)[1:]  # drop background=0
        masks = mask == obj_ids[:, None, None]  # shape [N, H, W]

        # Compute bounding boxes
        boxes_xyxy = []
        for m in masks:
            ys, xs = np.where(m)
            boxes_xyxy.append([xs.min(), ys.min(), xs.max(), ys.max()])

        boxes_xywh = []
        areas = []
        iscrowd = []

        for (x1,y1,x2,y2) in boxes_xyxy:
            w = x2 - x1
            h = y2 - y1
            boxes_xywh.append([x1, y1, w, h])
            areas.append(w * h)
            iscrowd.append(0)

        labels = [1] * len(boxes_xywh)

        annots = []
        for box, lab, area, crowd in zip(boxes_xywh, labels, areas, iscrowd):
            annots.append({
                "bbox": box,
                "category_id": int(lab),
                "area": float(area),
                "iscrowd": int(crowd),
            })

        target = {
            "image_id": idx,
            "annotations": annots
        }

        # resize to 800×800 for DETR
        if self.transforms:
            img = self.transforms(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders():
    transform = transforms.Resize((800,800))
    dataset = PennFudanDataset(DATA_ROOT, transforms=transform)
    # split 80/20
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_dl, val_dl
