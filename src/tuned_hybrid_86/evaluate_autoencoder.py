# -*- coding: utf-8 -*-
"""
evaluate_hybrid_tuned.py

Standalone evaluation for the tuned CNN-AE Hybrid model.
Loads best checkpoint from checkpoints_tuned/best.pth and
the best hyperparameters found by Optuna.

Run after autoencoder_hybrid_tuned.py has completed.
"""

import random
from pathlib import Path

from pre_process_data import PreProcessData
from datasets import CLASS_TO_IDX, IDX_TO_CLASS

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve, precision_recall_curve)


# ---------------------------------------------------------------------------
# Best hyperparameters from Optuna — update these if you re-run tuning
# ---------------------------------------------------------------------------

BEST_PARAMS = {
    "lr":           0.004896799190396437,
    "weight_decay": 0.0014918352243523305,
    "w_recon":      0.7966451488157834,
    "w_cls":        2.1519415990911366,
    "cls_dropout":  0.467893887526443,
    "score_alpha":  0.6901497433409689,
    "enc_channels": 64,
    "batch_size":   32,
    "slice_stride": 3,
}

BEST_CKPT = Path("checkpoints_tuned/best.pth")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE  = 128
JITTER_PX   = 6
SEED        = 6767

EXP2_EXPORT = Path("data/export_exp2/")
EXP2_CSV    = Path("data/labels_exp2.csv")

NORMAL_CLASSES  = {"TB", "TM"}
ANOMALY_CLASSES = {"FB", "FM"}
ANOMALY_IDX_SET = {CLASS_TO_IDX[c] for c in ANOMALY_CLASSES}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

assert BEST_CKPT.exists(), \
    f"No checkpoint at {BEST_CKPT} — run autoencoder_hybrid_tuned.py first."


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CTPatchDataset(Dataset):
    def __init__(self, export_root, labels_csv, patch_size=128,
                 augment=False, jitter_px=0, normal_classes=("TB", "TM")):
        self.export_root    = Path(export_root)
        self.patch_size     = patch_size
        self.augment        = augment
        self.jitter_px      = jitter_px
        self.normal_classes = set(normal_classes)
        self.half           = patch_size // 2
        self._vol_cache     = {}

        df_raw = pd.read_csv(labels_csv)
        valid  = [row.to_dict() for _, row in df_raw.iterrows()
                  if (self.export_root / str(row["uuid"]) / "volume.npz").exists()]
        self.df = pd.DataFrame(valid).reset_index(drop=True)
        print(f"Loaded {len(self.df)}/{len(df_raw)} rows")

    def _load_volume(self, uuid):
        if uuid not in self._vol_cache:
            self._vol_cache[uuid] = np.load(
                self.export_root / uuid / "volume.npz",
                allow_pickle=False)["volume"]
        return self._vol_cache[uuid]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        uuid  = str(row["uuid"])
        slc   = int(row["slice"])
        cls   = str(row["type"])
        label = CLASS_TO_IDX[cls]

        volume = self._load_volume(uuid)
        slc    = min(slc, volume.shape[0] - 1)
        img    = volume[slc].astype(np.float32)
        H, W   = img.shape

        if cls in self.normal_classes:
            cx = random.randint(self.half, W - self.half)
            cy = random.randint(self.half, H - self.half)
        else:
            cx = int(row.get("x", W // 2))
            cy = int(row.get("y", H // 2))
            cx = max(self.half, min(cx, W - self.half))
            cy = max(self.half, min(cy, H - self.half))

        x0, x1 = cx - self.half, cx + self.half
        y0, y1 = cy - self.half, cy + self.half
        patch  = img[y0:y1, x0:x1]
        if patch.shape != (self.patch_size, self.patch_size):
            patch = np.pad(patch,
                           ((max(0, -y0), max(0, y1 - H)),
                            (max(0, -x0), max(0, x1 - W))),
                           mode="reflect")

        p_min, p_max = patch.min(), patch.max()
        patch = (patch - p_min) / (p_max - p_min + 1e-8)

        return torch.from_numpy(patch).unsqueeze(0), \
               torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model (must match training architecture exactly)
# ---------------------------------------------------------------------------

enc_channels = BEST_PARAMS["enc_channels"]
cls_dropout  = BEST_PARAMS["cls_dropout"]


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch); self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res(x); return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.res = ResidualBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g /= g.sum()
        self.register_buffer("kernel",
            (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0))

    def _compute(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        pad = self.window_size // 2
        k   = self.kernel.expand(x.size(1), 1, -1, -1)
        mx  = F.conv2d(x,   k, padding=pad, groups=x.size(1))
        my  = F.conv2d(y,   k, padding=pad, groups=x.size(1))
        sx  = F.conv2d(x*x, k, padding=pad, groups=x.size(1)) - mx**2
        sy  = F.conv2d(y*y, k, padding=pad, groups=x.size(1)) - my**2
        sxy = F.conv2d(x*y, k, padding=pad, groups=x.size(1)) - mx*my
        return ((2*mx*my+C1)*(2*sxy+C2)) / ((mx**2+my**2+C1)*(sx+sy+C2)+1e-8)

    def forward(self, x, y): return 1.0 - self._compute(x, y).mean()

    def per_sample(self, x, y):
        return 1.0 - self._compute(x, y).mean(dim=[1, 2, 3])


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1      = DownBlock(1,   32)
        self.down2      = DownBlock(32,  64)
        self.down3      = DownBlock(64,  enc_channels)
        self.bottleneck = ResidualBlock(enc_channels, enc_channels)

    def forward(self, x):
        x, s1 = self.down1(x); x, s2 = self.down2(x); x, s3 = self.down3(x)
        return self.bottleneck(x), [s1, s2, s3]


class DecoderHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3      = UpBlock(enc_channels, enc_channels, 64)
        self.up2      = UpBlock(64,           64,           32)
        self.up1      = UpBlock(32,           32,           16)
        self.out_conv = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, z, skips):
        s1, s2, s3 = skips
        return self.out_conv(self.up1(self.up2(self.up3(z, s3), s2), s1))


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_channels, 64), nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(32, 1),
        )

    def forward(self, z): return self.mlp(self.gap(z))


class CNNAEHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = SharedEncoder()
        self.decoder    = DecoderHead()
        self.classifier = ClassifierHead()

    def forward(self, x):
        z, skips = self.encoder(x)
        return self.decoder(z, skips), self.classifier(z)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model = CNNAEHybrid().to(DEVICE)
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))
model.eval()
print(f"Loaded model from {BEST_CKPT}\n")

ssim_fn = SSIMLoss().to(DEVICE)

# ---------------------------------------------------------------------------
# Dataset + loader
# ---------------------------------------------------------------------------

test_ds     = CTPatchDataset(EXP2_EXPORT, EXP2_CSV, PATCH_SIZE,
                              augment=False, jitter_px=0,
                              normal_classes=tuple(NORMAL_CLASSES))
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"Test samples: {len(test_ds)}")
print(pd.read_csv(EXP2_CSV)["type"].value_counts(), "\n")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

all_true        = []
all_cls_probs   = []
all_ssim_scores = []
all_class_names = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        recon, logit = model(xb)
        probs = torch.sigmoid(logit.squeeze(1)).cpu().numpy()
        ssims = ssim_fn.per_sample(recon, xb).cpu().numpy()
        for i in range(len(yb)):
            label = yb[i].item()
            all_true.append(1 if label in ANOMALY_IDX_SET else 0)
            all_cls_probs.append(probs[i])
            all_ssim_scores.append(ssims[i])
            all_class_names.append(IDX_TO_CLASS[label])

all_true        = np.array(all_true)
all_cls_probs   = np.array(all_cls_probs)
all_ssim_scores = np.array(all_ssim_scores)

# Blend classifier + SSIM
score_alpha  = BEST_PARAMS["score_alpha"]
ssim_norm    = ((all_ssim_scores - all_ssim_scores.min()) /
                (all_ssim_scores.max() - all_ssim_scores.min() + 1e-8))
final_scores = score_alpha * all_cls_probs + (1 - score_alpha) * ssim_norm

# Optimal threshold
precisions, recalls, pr_thresh = precision_recall_curve(all_true, final_scores)
f1s         = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx    = np.argmax(f1s)
best_thresh = pr_thresh[best_idx] if best_idx < len(pr_thresh) else pr_thresh[-1]
all_pred    = (final_scores >= best_thresh).astype(int)

auc = roc_auc_score(all_true, final_scores)
fpr, tpr, roc_thresh = roc_curve(all_true, final_scores)
acc = (all_true == all_pred).mean() * 100

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print("=" * 60)
print("  EVALUATION — CNN-AE HYBRID (TUNED)")
print("  Normal: TB, TM  |  Anomaly: FB, FM")
print("=" * 60)
print(f"\nROC-AUC:              {auc:.4f}  (baseline: 0.8426)")
print(f"Optimal threshold:    {best_thresh:.4f}")
print(f"Accuracy:             {acc:.2f}%\n")
print(classification_report(all_true, all_pred,
                             target_names=["Normal (TB/TM)", "Anomaly (FB/FM)"],
                             zero_division=0))

print("Classifier probability per class:")
for cls in ["TB", "TM", "FB", "FM"]:
    mask = np.array(all_class_names) == cls
    if mask.sum() == 0: continue
    p = all_cls_probs[mask]
    print(f"  [{cls}] mean={p.mean():.4f}  std={p.std():.4f}  n={mask.sum()}")
print("=" * 60)

print("\nBest hyperparameters:")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

colors = {"TB": "steelblue", "TM": "cornflowerblue",
          "FB": "tomato",    "FM": "salmon"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"CNN-AE Hybrid (Tuned) — AUC={auc:.4f}  baseline=0.8426",
             fontsize=13)

# Score distribution
for cls in ["TB", "TM", "FB", "FM"]:
    mask = np.array(all_class_names) == cls
    if mask.sum() == 0: continue
    axes[0].hist(all_cls_probs[mask], bins=20, alpha=0.6,
                 label=cls, color=colors[cls])
axes[0].axvline(0.5, color="black", linestyle="--", lw=1.5, label="0.5")
axes[0].set_xlabel("Classifier Probability (anomaly)")
axes[0].set_ylabel("Count")
axes[0].set_title("Score Distribution by Class")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# ROC curve
axes[1].plot(fpr, tpr, lw=2, color="darkorange", label=f"AUC={auc:.4f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
op_idx = np.argmin(np.abs(roc_thresh - best_thresh))
axes[1].scatter(fpr[op_idx], tpr[op_idx], s=80, zorder=5,
                color="red", label="Operating point")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(all_true, all_pred)
ConfusionMatrixDisplay(cm,
    display_labels=["Normal\n(TB/TM)", "Anomaly\n(FB/FM)"]
).plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("evaluation_tuned.png", dpi=150)
plt.show()
print("\nSaved → evaluation_tuned.png")


# ---------------------------------------------------------------------------
# Reconstruction grid
# ---------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    xb, yb = next(iter(test_loader))
    xb     = xb.to(DEVICE)
    recon, logit = model(xb)
    probs  = torch.sigmoid(logit.squeeze(1))

n_show = min(8, xb.size(0))
fig, axes = plt.subplots(3, n_show, figsize=(2.2 * n_show, 7))

for i in range(n_show):
    orig      = xb[i].cpu().squeeze().numpy()
    rec       = recon[i].cpu().squeeze().numpy()
    error     = np.abs(orig - rec)
    cls_name  = IDX_TO_CLASS[yb[i].item()]
    prob      = probs[i].item()
    predicted = "Anomaly" if prob >= 0.5 else "Normal"
    is_anom   = cls_name in ANOMALY_CLASSES
    correct   = predicted == ("Anomaly" if is_anom else "Normal")

    axes[0, i].imshow(orig, cmap="gray")
    axes[0, i].set_title(f"True: {cls_name}", fontsize=8)
    axes[0, i].axis("off")

    axes[1, i].imshow(rec, cmap="gray")
    axes[1, i].set_title(
        f"{'✓' if correct else '✗'} {predicted}\np={prob:.2f}",
        fontsize=7, color="green" if correct else "red")
    axes[1, i].axis("off")

    axes[2, i].imshow(error, cmap="hot")
    axes[2, i].axis("off")

axes[0, 0].set_ylabel("Original",       fontsize=9)
axes[1, 0].set_ylabel("Reconstruction", fontsize=9)
axes[2, 0].set_ylabel("Error Map",      fontsize=9)

plt.suptitle("CNN-AE Hybrid (Tuned) — Reconstruction Grid",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig("reconstruction_grid_tuned.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → reconstruction_grid_tuned.png")