import torch
from tqdm import tqdm
from config import DEVICE, EPOCHS, LR
from dataset import get_dataloaders
from model import DETRModel

def train():
    train_dl, val_dl = get_dataloaders()
    model = DETRModel(device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for images, targets in pbar:
            outputs = model(images, targets)
            loss = model.compute_loss(outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{epoch_loss/(pbar.n+1):.4f}"})
        pbar.close()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dl:
                images  = list(img.to(DEVICE) for img in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                outputs = model(images, targets)
                val_loss += model.compute_loss(outputs).item()
        print(f"Validation Loss: {val_loss/len(val_dl):.4f}\n")

    # Save a checkpoint for later visualization/inference
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": EPOCHS
    }
    torch.save(ckpt, "detr_pennfudan.pth")
    print("Training complete! Checkpoint saved to detr_pennfudan.pth")

if __name__ == "__main__":
    train()
