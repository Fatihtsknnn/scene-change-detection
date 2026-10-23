import os
import glob
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

class LevirDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.a_paths = sorted(glob.glob(os.path.join(root, "A", split + "*.png")))

    def __len__(self):
        return len(self.a_paths)


    def __getitem__(self, idx):
        # 1) yollar
        a_path = self.a_paths[idx]
        b_path = a_path.replace("/A/", "/B/")
        label_path = a_path.replace("/A/", "/label/")

        # 2) oku (cv2 görselleri BGR okur, modele RGB lazım -> çeviriyoruz)
        a = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                # 3) kucult: 1024 -> 256 (GPU icin hafif, hizli)
        a = cv2.resize(a, (256, 256))
        b = cv2.resize(b, (256, 256))
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 4) normalize (0-255 -> 0-1) + kanal sirasini degistir (HWC -> CHW)
        a = a.astype(np.float32) / 255.0
        b = b.astype(np.float32) / 255.0
        a = a.transpose(2, 0, 1)   # 256x256x3 -> 3x256x256
        b = b.transpose(2, 0, 1)

        # 5) A ve B'yi ust uste yapistir -> 6 kanal (isin kalbi burasi)
        x = np.concatenate([a, b], axis=0)   # 6x256x256

        # 6) maske -> 0/1, tek kanal
        label = (label > 127).astype(np.float32)   # beyaz=1 (degisti), siyah=0
        label = label[np.newaxis, :, :]            # 256x256 -> 1x256x256

        # tensore cevir ve dondur
        x = torch.from_numpy(x)
        label = torch.from_numpy(label)
        return x, label

if __name__ == "__main__":
    ds = LevirDataset("/home/fatih/levir-cd", "train")
    print("ornek sayisi:", len(ds))
    x, y = ds[0]
    print("girdi shape:", x.shape, "| maske shape:", y.shape)

        