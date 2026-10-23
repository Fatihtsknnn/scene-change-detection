import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from dataset import LevirDataset
import os
ROOT = os.environ.get("LEVIR_ROOT", "/home/fatih/levir-cd")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("cihaz:", device)

# veri
train_ds = LevirDataset(ROOT, "train")
val_ds   = LevirDataset(ROOT, "val")
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

# model: 6 kanal giris, 1 kanal cikis
model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=6, classes=1).to(device)

# loss + optimizer
loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# tensorboard
writer = SummaryWriter("runs/sahne_degisim")

epochs = 20
best_val = 999

for epoch in range(epochs):
    # ---- EGITIM ----
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        tahmin = model(x)
        loss = loss_fn(tahmin, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ---- DOGRULAMA ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            tahmin = model(x)
            val_loss += loss_fn(tahmin, y).item()
    val_loss /= len(val_loader)

    # ---- KAYIT + EKRAN ----
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    print(f"epoch {epoch+1}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

    # ---- EN IYIYI KAYDET ----
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "sahne_model.pt")
        print("  -> en iyi model kaydedildi")

writer.close()