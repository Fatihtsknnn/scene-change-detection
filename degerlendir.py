import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import LevirDataset
import os
ROOT = os.environ.get("LEVIR_ROOT", "/home/fatih/levir-cd")

device = "cuda" if torch.cuda.is_available() else "cpu"

test_ds = LevirDataset(ROOT, "test")
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

# modeli kur ve kayitli agirliklari yukle
model = smp.Unet("resnet34", encoder_weights=None, in_channels=6, classes=1).to(device)
model.load_state_dict(torch.load("sahne_model.pt", map_location=device))
model.eval()

tps, fps, fns, tns = [], [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        tahmin = model(x)
        prob = torch.sigmoid(tahmin)          # ham skor -> 0-1 olasilik
        tp, fp, fn, tn = smp.metrics.get_stats(
            prob, y.long(), mode="binary", threshold=0.5
        )
        tps.append(tp); fps.append(fp); fns.append(fn); tns.append(tn)

# tum test setini birlestir
tp = torch.cat(tps); fp = torch.cat(fps)
fn = torch.cat(fns); tn = torch.cat(tns)

iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
f1  = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

print(f"Test IoU : {iou:.4f}")
print(f"Test F1  : {f1:.4f}  (Dice)")