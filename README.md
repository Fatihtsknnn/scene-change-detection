# Scene Change Detection — LEVIR-CD

Bitemporal (before/after) satellite change detection with a U-Net.
Given two aerial images of the same location taken at different times, the model
predicts a binary mask of the regions that changed (mainly new / removed buildings).

Developed as a take-home task for TURKAI.

## Approach
Change detection is framed as **binary segmentation**:
- The *before* image (A, 3 channels) and *after* image (B, 3 channels) are
  concatenated along the channel axis → a **6-channel** input.
- A **U-Net** (ResNet-34 encoder, ImageNet-pretrained) maps that 6-channel input
  to a **1-channel** change mask.
- Trained with **Dice loss**; predictions are sigmoid-thresholded at 0.5.

Rationale: concatenating the bitemporal pair lets a standard, proven segmentation
network learn the difference directly — simple, fully in PyTorch, and reusing a
well-understood architecture rather than a custom one.

## Dataset
[LEVIR-CD](https://github.com/justchenhao/STANet) — building change detection,
1024×1024 bitemporal pairs with binary change masks.
- Train: 445 · Val: 64 · Test: 128
- Layout: `A/` (before), `B/` (after), `label/` (mask); a pair shares its filename.
- Images are resized to 256×256 at load time.

## Results (test set, 128 pairs)
| Metric | Score |
|--------|-------|
| F1 / Dice | **0.739** |
| IoU | **0.586** |

Qualitative examples (before · after · prediction · ground truth): see `sonuc.png`.

## Project structure
dataset.py # LevirDataset: loads A+B (6ch) and the change mask
egit.py # training + validation + TensorBoard logging + best checkpoint
degerlendir.py # evaluation on the test set (IoU, F1/Dice)
gorsel.py # qualitative visualization -> sonuc.png
requirements.txt
Dockerfile
sahne_model.pt # trained weights (produced by egit.py)


## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Download LEVIR-CD so that `A/`, `B/`, `label/` sit under one folder, then point the
scripts at it via the `LEVIR_ROOT` environment variable
(default: `/home/fatih/levir-cd`).

## Usage
```bash
python egit.py              # train
tensorboard --logdir runs   # view loss curves
python degerlendir.py       # evaluate on test set
python gorsel.py            # produce sonuc.png
```

## Docker
```bash
docker build -t sahne-degisim .
docker run --rm -e LEVIR_ROOT=/data -v ~/levir-cd:/data \
  -v $(pwd)/sahne_model.pt:/app/sahne_model.pt sahne-degisim
```

## Limitations & future work
- Images are downsampled 1024→256, so fine boundaries are lost — predicted masks
  are blobbier than ground truth.
- Small, densely packed buildings merge or get missed (strong class imbalance:
  changed pixels are a small fraction of each scene).
- Next steps: train at full resolution with tiling; a stronger / siamese
  change-detection architecture (feature-difference or attention); class-balanced
  loss; data augmentation.

## Tech stack
Python · PyTorch · segmentation-models-pytorch · OpenCV · TensorBoard · Docker