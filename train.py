import torch, time, datetime, argparse, os
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import torchvision
from tqdm import tqdm

import config as C
from dataset import PennFudanDataset, collate_fn
from model import get_model, get_device

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}", leave=False):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
    return epoch_loss / len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total = correct = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            # Very cheap metric: did we detect at least 1 pedestrian?
            has_ped = (out["scores"] > 0.5).any().item()
            correct += int(has_ped)
            total += 1
    return correct / total


def main(args):
    device = get_device(args.device)

    full_ds = PennFudanDataset(C.DATA_ROOT, transforms=True)
    val_len = int(0.2 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_len, val_len])
    train_ds.dataset.transforms_flag = True   # make sure augmentation ON
    val_ds.dataset.transforms_flag = False  # no aug for val

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=C.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=C.NUM_WORKERS, collate_fn=collate_fn)

    model = get_model(num_classes=C.NUM_CLASSES)
    model.to(device)

    # Param groups give backbone lower LR
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n
                                                  and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n
                                                  and p.requires_grad],
         "lr": C.LR_BACKBONE},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    lr_scheduler = StepLR(optimizer, step_size=C.LR_DROP_EPOCHS, gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, C.EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        acc  = evaluate(model, val_loader, device)
        lr_scheduler.step()

        print(f"[{epoch:02}/{C.EPOCHS}] loss={loss:.4f}  val-acc={acc*100:.1f}%")
        ckpt_path = C.CKPT_DIR / f"epoch{epoch:02}.pth"
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "acc": acc}, ckpt_path)

        # Save best
        if acc > best_acc:
            torch.save(model.state_dict(), C.CKPT_DIR / "best.pth")
            best_acc = acc

    print(f"Training finished. Best val acc: {best_acc*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="'cuda'|'cpu' (default: auto)")
    args = ap.parse_args()
    main(args)