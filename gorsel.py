import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from dataset import LevirDataset
import os
ROOT = os.environ.get("LEVIR_ROOT", "/home/fatih/levir-cd")

device = "cuda" if torch.cuda.is_available() else "cpu"

test_ds = LevirDataset(ROOT, "test")

model = smp.Unet("resnet34", encoder_weights=None, in_channels=6, classes=1).to(device)
model.load_state_dict(torch.load("sahne_model.pt", map_location=device))
model.eval()

ornekler = [0, 50]   # gosterecegimiz 2 ornek
fig, axes = plt.subplots(len(ornekler), 4, figsize=(14, 7))

for satir, idx in enumerate(ornekler):
    x, y = test_ds[idx]
    with torch.no_grad():
        tahmin = model(x.unsqueeze(0).to(device))      # tek ornegi batch yap
        pred = (torch.sigmoid(tahmin) > 0.5).float()

    a = x[:3].permute(1, 2, 0).numpy()     # ilk 3 kanal = ONCE
    b = x[3:].permute(1, 2, 0).numpy()     # son 3 kanal = SONRA
    pred = pred.squeeze().cpu().numpy()    # tahmin maskesi
    gercek = y.squeeze().numpy()           # gercek maske

    axes[satir, 0].imshow(a);                 axes[satir, 0].set_title("Once (A)")
    axes[satir, 1].imshow(b);                 axes[satir, 1].set_title("Sonra (B)")
    axes[satir, 2].imshow(pred, cmap="gray"); axes[satir, 2].set_title("Tahmin")
    axes[satir, 3].imshow(gercek, cmap="gray"); axes[satir, 3].set_title("Gercek")
    for j in range(4):
        axes[satir, j].axis("off")

plt.tight_layout()
plt.savefig("sonuc.png", dpi=120)
print("sonuc.png kaydedildi")