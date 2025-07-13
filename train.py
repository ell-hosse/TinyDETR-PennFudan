import torch
import argparse
from torch.optim import AdamW
from tqdm import tqdm
from model import get_model, get_device
from dataset import get_dataloaders
import config as C


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main(args):
    device = get_device(args.device)
    model = get_model(num_classes=C.NUM_CLASSES, pretrained=True).to(device)
    train_dl, val_dl = get_dataloaders()

    optimizer = AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

    best_val_loss = float("inf")

    for epoch in range(1, C.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, device, epoch)
        val_loss = evaluate(model, val_dl, device)

        print(f"[{epoch:02}/{C.EPOCHS}] Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), C.CKPT_DIR / "best.pth")
            best_val_loss = val_loss
            print(f"Best model saved at epoch {epoch}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto)")
    args = parser.parse_args()
    main(args)
