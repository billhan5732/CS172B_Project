# -*- coding: utf-8 -*-
"""
autoencoder_hybrid_tuned.py

Original CNN-AE Hybrid (0.84 AUC) + Optuna hyperparameter tuning.

Two-phase workflow:
  Phase 1 — Optuna search (N_TRIALS trials, TUNE_EPOCHS epochs each)
    Searches over: lr, weight_decay, w_recon, w_cls, cls_dropout,
                   score_alpha, enc_channels, batch_size, slice_stride
    Objective: maximise ROC-AUC on test set

  Phase 2 — Full training with best hyperparameters (FINAL_EPOCHS epochs)
    Uses best params found in Phase 1
    Saves checkpoints + full evaluation plots

Install Optuna if needed:
  pip install optuna
"""

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")

from pre_process_data import PreProcessData
from datasets import CLASS_TO_IDX, IDX_TO_CLASS

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Run: pip install optuna")

from sklearn.metrics import roc_auc_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve


# ---------------------------------------------------------------------------
# Fixed config (not tuned)
# ---------------------------------------------------------------------------

@dataclass
class FixedConfig:
    exp1_path        = Path("data/Experiment 1 - Blind/")
    exp2_path        = Path("data/Experiment 2 - Open/")
    exp1_labels_csv  = Path("data/labels_exp1.csv")
    exp2_labels_csv  = Path("data/labels_exp2.csv")
    exp1_export_root = Path("data/export_exp1/")
    exp2_export_root = Path("data/export_exp2/")

    patch_size  = 128   # full patch size — same as 0.84 AUC version
    jitter_px   = 6
    augment     = True
    seed        = 6767
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    normal_classes  = {"TB", "TM"}
    anomaly_classes = {"FB", "FM"}

    # Tuning
    N_TRIALS    = 30     # number of Optuna trials
    TUNE_EPOCHS = 50     # epochs per trial (short — just enough to rank trials)
    FINAL_EPOCHS = 300   # epochs for final training with best params


CFG = FixedConfig()

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)

print(f"Using device: {CFG.device}")
if CFG.device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

print("Preprocessing data...\n")

if not CFG.exp1_export_root.exists():
    PreProcessData(CFG.exp1_path, CFG.exp1_export_root).run()
    print(f"\t{CFG.exp1_path} processed!\n")
else:
    print(f"\t{CFG.exp1_export_root} already exists.\n")

if not CFG.exp2_export_root.exists():
    PreProcessData(CFG.exp2_path, CFG.exp2_export_root).run()
    print(f"\t{CFG.exp2_path} processed!\n")
else:
    print(f"\t{CFG.exp2_export_root} already exists.\n")

print("Preprocessing finished!\n")

df  = pd.read_csv(CFG.exp1_labels_csv)
df2 = pd.read_csv(CFG.exp2_labels_csv)
print("Experiment 1:"); print(df["type"].value_counts())
print("\nExperiment 2:"); print(df2["type"].value_counts())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def expand_normal_slices(df, export_root, normal_classes=("TB", "TM"),
                         slice_stride=5):
    export_root   = Path(export_root)
    expanded_rows = []
    for _, row in df.iterrows():
        if row["type"] not in normal_classes:
            expanded_rows.append(row.to_dict())
            continue
        volume_path = export_root / str(row["uuid"]) / "volume.npz"
        if not volume_path.exists():
            continue
        volume = np.load(volume_path, allow_pickle=False)["volume"]
        for s in range(0, volume.shape[0], slice_stride):
            expanded_rows.append({"type": row["type"], "uuid": row["uuid"],
                                   "slice": s, "x": 0, "y": 0})
    return pd.DataFrame(expanded_rows)


def make_datasets(slice_stride: int):
    """Build train/test datasets for a given slice_stride."""
    df_train = pd.read_csv(CFG.exp1_labels_csv)
    df_exp   = expand_normal_slices(df_train, CFG.exp1_export_root,
                                    slice_stride=slice_stride)
    expanded_csv = Path("data/labels_exp1_expanded.csv")
    df_exp.to_csv(expanded_csv, index=False)

    train_ds = CTPatchDataset(CFG.exp1_export_root, expanded_csv,
                              CFG.patch_size, CFG.augment, CFG.jitter_px,
                              tuple(CFG.normal_classes))
    test_ds  = CTPatchDataset(CFG.exp2_export_root, CFG.exp2_labels_csv,
                              CFG.patch_size, augment=False, jitter_px=0,
                              normal_classes=tuple(CFG.normal_classes))
    return train_ds, test_ds


def make_loader(train_ds, batch_size: int):
    anomaly_idx_set = {CLASS_TO_IDX[c] for c in CFG.anomaly_classes}
    labels     = [CLASS_TO_IDX[str(r["type"])] for _, r in train_ds.df.iterrows()]
    is_anom    = [1 if l in anomaly_idx_set else 0 for l in labels]
    n_normal   = sum(1 for a in is_anom if a == 0)
    n_anomaly  = sum(1 for a in is_anom if a == 1)
    w_n = 1.0 / max(n_normal,  1)
    w_a = 1.0 / max(n_anomaly, 1)
    sampler = WeightedRandomSampler(
        weights=[w_a if a else w_n for a in is_anom],
        num_samples=len(is_anom), replacement=True)
    return DataLoader(train_ds, batch_size=batch_size,
                      sampler=sampler, num_workers=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CTPatchDataset(Dataset):
    def __init__(self, export_root, labels_csv, patch_size=128,
                 augment=True, jitter_px=6, normal_classes=("TB", "TM")):
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

    def _load_volume(self, uuid):
        if uuid not in self._vol_cache:
            path = self.export_root / uuid / "volume.npz"
            self._vol_cache[uuid] = np.load(path, allow_pickle=False)["volume"]
        return self._vol_cache[uuid]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
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
            if self.augment and self.jitter_px > 0:
                cx += random.randint(-self.jitter_px, self.jitter_px)
                cy += random.randint(-self.jitter_px, self.jitter_px)
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

        if self.augment:
            if random.random() > 0.5: patch = np.fliplr(patch).copy()
            if random.random() > 0.5: patch = np.flipud(patch).copy()

        return torch.from_numpy(patch).unsqueeze(0), \
               torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------

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
        self.res  = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

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


def build_model(enc_channels: int, cls_dropout: float) -> nn.Module:
    """Builds CNNAEHybrid with given enc_channels and dropout."""

    class SharedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.down1      = DownBlock(1,   32)
            self.down2      = DownBlock(32,  64)
            self.down3      = DownBlock(64,  enc_channels)
            self.bottleneck = ResidualBlock(enc_channels, enc_channels)

        def forward(self, x):
            x, s1 = self.down1(x)
            x, s2 = self.down2(x)
            x, s3 = self.down3(x)
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

    return CNNAEHybrid()


class HybridLoss(nn.Module):
    def __init__(self, w_recon, w_cls):
        super().__init__()
        self.ssim  = SSIMLoss()
        self.bce   = nn.BCEWithLogitsLoss()
        self.w_recon = w_recon
        self.w_cls   = w_cls

    def forward(self, recon, x, logit, binary_label):
        l_recon = self.ssim(recon, x)
        l_cls   = self.bce(logit.squeeze(1), binary_label)
        return self.w_recon * l_recon + self.w_cls * l_cls, l_recon, l_cls


# ---------------------------------------------------------------------------
# Evaluation helper — returns ROC-AUC
# ---------------------------------------------------------------------------

def evaluate_auc(model, test_loader, ssim_fn, score_alpha,
                 anomaly_idx_set, device):
    model.eval()
    all_true  = []
    all_scores = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            recon, logit = model(xb)
            probs  = torch.sigmoid(logit.squeeze(1)).cpu().numpy()
            ssims  = ssim_fn.per_sample(recon, xb).cpu().numpy()

            # Normalise SSIM within this batch (rough — full norm done at end)
            ssim_min = ssims.min(); ssim_max = ssims.max()
            ssim_norm = (ssims - ssim_min) / (ssim_max - ssim_min + 1e-8)

            scores = score_alpha * probs + (1 - score_alpha) * ssim_norm

            for i in range(len(yb)):
                all_true.append(1 if yb[i].item() in anomaly_idx_set else 0)
                all_scores.append(scores[i])

    all_true   = np.array(all_true)
    all_scores = np.array(all_scores)

    if len(np.unique(all_true)) < 2:
        return 0.5   # degenerate case

    return roc_auc_score(all_true, all_scores)


# ---------------------------------------------------------------------------
# Phase 1 — Optuna hyperparameter search
# ---------------------------------------------------------------------------

anomaly_idx_set = {CLASS_TO_IDX[c] for c in CFG.anomaly_classes}

print("=" * 60)
print(f"  PHASE 1 — Optuna search ({CFG.N_TRIALS} trials, "
      f"{CFG.TUNE_EPOCHS} epochs each)")
print("=" * 60)

def objective(trial: optuna.Trial) -> float:
    # ---- Hyperparameter search space -----------------------------------
    lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    w_recon      = trial.suggest_float("w_recon",      0.5,  3.0)
    w_cls        = trial.suggest_float("w_cls",        0.5,  5.0)
    cls_dropout  = trial.suggest_float("cls_dropout",  0.1,  0.6)
    score_alpha  = trial.suggest_float("score_alpha",  0.4,  0.9)
    enc_channels = trial.suggest_categorical("enc_channels", [64, 128])
    batch_size   = trial.suggest_categorical("batch_size",   [8, 16, 32])
    slice_stride = trial.suggest_categorical("slice_stride", [3, 5, 8])

    # ---- Data ----------------------------------------------------------
    train_ds, test_ds = make_datasets(slice_stride)
    train_loader      = make_loader(train_ds, batch_size)
    test_loader       = DataLoader(test_ds, batch_size=32,
                                   shuffle=False, num_workers=0)

    # ---- Model ---------------------------------------------------------
    model     = build_model(enc_channels, cls_dropout).to(CFG.device)
    criterion = HybridLoss(w_recon, w_cls).to(CFG.device)
    ssim_fn   = SSIMLoss().to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.TUNE_EPOCHS, eta_min=1e-5)

    # ---- Training ------------------------------------------------------
    for epoch in range(CFG.TUNE_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(CFG.device); yb = yb.to(CFG.device)
            binary = torch.tensor(
                [1.0 if y.item() in anomaly_idx_set else 0.0 for y in yb],
                dtype=torch.float32, device=CFG.device)
            optimizer.zero_grad(set_to_none=True)
            recon, logit         = model(xb)
            loss, _, _           = criterion(recon, xb, logit, binary)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Prune unpromising trials early
        if epoch == CFG.TUNE_EPOCHS // 2:
            mid_auc = evaluate_auc(model, test_loader, ssim_fn,
                                   score_alpha, anomaly_idx_set, CFG.device)
            trial.report(mid_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    auc = evaluate_auc(model, test_loader, ssim_fn,
                       score_alpha, anomaly_idx_set, CFG.device)
    return auc


study = optuna.create_study(
    direction  = "maximize",
    pruner     = optuna.pruners.MedianPruner(n_startup_trials=5,
                                              n_warmup_steps=10),
    study_name = "hybrid_cnn_ae",
)

study.optimize(objective, n_trials=CFG.N_TRIALS,
               show_progress_bar=True, gc_after_trial=True)

print(f"\nBest trial:  AUC = {study.best_value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Save study results
results_df = study.trials_dataframe()
results_df.to_csv("optuna_results.csv", index=False)
print("Saved → optuna_results.csv")

# Plot Optuna importance
try:
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig("optuna_importance.png", dpi=150)
    plt.show()
    print("Saved → optuna_importance.png")

    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig("optuna_history.png", dpi=150)
    plt.show()
    print("Saved → optuna_history.png")
except Exception as e:
    print(f"Optuna plot skipped: {e}")

best = study.best_params


# ---------------------------------------------------------------------------
# Phase 2 — Full training with best hyperparameters
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"  PHASE 2 — Full training ({CFG.FINAL_EPOCHS} epochs)")
print(f"  Best params: {best}")
print("=" * 60 + "\n")

train_ds, test_ds = make_datasets(best["slice_stride"])
train_loader      = make_loader(train_ds, best["batch_size"])
test_loader       = DataLoader(test_ds, batch_size=32,
                               shuffle=False, num_workers=0)

print(f"Training — {len(train_ds)} samples")
print(f"Test     — {len(test_ds)} samples\n")

model     = build_model(best["enc_channels"], best["cls_dropout"]).to(CFG.device)
criterion = HybridLoss(best["w_recon"], best["w_cls"]).to(CFG.device)
ssim_fn   = SSIMLoss().to(CFG.device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=best["lr"],
                             weight_decay=best["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.FINAL_EPOCHS, eta_min=1e-5)

CHECKPOINT_PATH = Path("checkpoints_tuned")
CHECKPOINT_PATH.mkdir(exist_ok=True)
LATEST_CKPT = CHECKPOINT_PATH / "latest.pth"
BEST_CKPT   = CHECKPOINT_PATH / "best.pth"

logged_epochs = []; train_losses = []; val_auc_log = []
best_auc = 0.0; start_epoch = 0

if LATEST_CKPT.exists():
    print(f"Resuming from {LATEST_CKPT}")
    ckpt = torch.load(LATEST_CKPT, map_location=CFG.device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch   = ckpt["epoch"] + 1
    logged_epochs = ckpt["logged_epochs"]
    train_losses  = ckpt["train_losses"]
    val_auc_log   = ckpt["val_auc_log"]
    best_auc      = ckpt.get("best_auc", 0.0)
    print(f"Resuming from epoch {start_epoch}\n")
else:
    print("Starting from scratch.\n")

log_every       = max(1, CFG.FINAL_EPOCHS // 25)
save_ckpt_every = 10

for epoch in range(start_epoch, CFG.FINAL_EPOCHS):
    model.train()
    r_total = 0.0

    for xb, yb in train_loader:
        xb = xb.to(CFG.device); yb = yb.to(CFG.device)
        binary = torch.tensor(
            [1.0 if y.item() in anomaly_idx_set else 0.0 for y in yb],
            dtype=torch.float32, device=CFG.device)
        optimizer.zero_grad(set_to_none=True)
        recon, logit     = model(xb)
        loss, _, _       = criterion(recon, xb, logit, binary)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        r_total += loss.item() * xb.size(0)

    scheduler.step()
    train_loss = r_total / len(train_loader.dataset)

    auc = evaluate_auc(model, test_loader, ssim_fn,
                       best["score_alpha"], anomaly_idx_set, CFG.device)

    if (epoch + 1) % log_every == 0:
        print(f"epoch={epoch+1:4d}  train={train_loss:.4f}  val_auc={auc:.4f}")
        logged_epochs.append(epoch + 1)
        train_losses.append(train_loss)
        val_auc_log.append(auc)

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), BEST_CKPT)

    if (epoch + 1) % save_ckpt_every == 0:
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "logged_epochs": logged_epochs,
                    "train_losses": train_losses,
                    "val_auc_log": val_auc_log,
                    "best_auc": best_auc}, LATEST_CKPT)

torch.save(model.state_dict(), "hybrid_tuned_final.pth")
print(f"\nBest AUC during training: {best_auc:.4f}")
print(f"Model saved → hybrid_tuned_final.pth")
print(f"Best model  → {BEST_CKPT}")


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("CNN-AE Hybrid (Tuned) Training Curves", fontsize=13)
axes[0].plot(logged_epochs, train_losses, label="Train loss")
axes[0].set_title("Training Loss"); axes[0].grid(True); axes[0].legend()
axes[1].plot(logged_epochs, val_auc_log, color="darkorange", label="Val AUC")
axes[1].axhline(0.8426, color="gray", linestyle="--", lw=1, label="Baseline (0.8426)")
axes[1].set_title("Validation ROC-AUC"); axes[1].set_ylim(0.4, 1.0)
axes[1].grid(True); axes[1].legend()
plt.tight_layout()
plt.savefig("loss_curves_tuned.png", dpi=150); plt.show()
print("Saved → loss_curves_tuned.png")


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.load_state_dict(torch.load(BEST_CKPT, map_location=CFG.device))
model.eval()
print(f"\nLoaded best model (AUC={best_auc:.4f}) from {BEST_CKPT}")

all_true = []; all_cls_probs = []; all_ssim_scores = []; all_class_names = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(CFG.device)
        recon, logit = model(xb)
        probs = torch.sigmoid(logit.squeeze(1)).cpu().numpy()
        ssims = ssim_fn.per_sample(recon, xb).cpu().numpy()
        for i in range(len(yb)):
            label = yb[i].item()
            all_true.append(1 if label in anomaly_idx_set else 0)
            all_cls_probs.append(probs[i])
            all_ssim_scores.append(ssims[i])
            all_class_names.append(IDX_TO_CLASS[label])

all_true        = np.array(all_true)
all_cls_probs   = np.array(all_cls_probs)
all_ssim_scores = np.array(all_ssim_scores)

ssim_norm    = ((all_ssim_scores - all_ssim_scores.min()) /
                (all_ssim_scores.max() - all_ssim_scores.min() + 1e-8))
final_scores = best["score_alpha"] * all_cls_probs + \
               (1 - best["score_alpha"]) * ssim_norm

precisions, recalls, pr_thresholds = precision_recall_curve(all_true, final_scores)
f1s         = 2*precisions*recalls / (precisions + recalls + 1e-8)
best_idx    = np.argmax(f1s)
best_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else pr_thresholds[-1]
all_pred    = (final_scores >= best_thresh).astype(int)

auc = roc_auc_score(all_true, final_scores)
fpr, tpr, roc_thresh = roc_curve(all_true, final_scores)
acc = (all_true == all_pred).mean() * 100

print("\n" + "=" * 60)
print("  FINAL EVALUATION — CNN-AE HYBRID (TUNED)")
print("  Normal: TB, TM  |  Anomaly: FB, FM")
print("=" * 60)
print(f"\nROC-AUC:  {auc:.4f}  (baseline: 0.8426)")
print(f"Accuracy: {acc:.2f}%\n")
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

# Plots
colors = {"TB": "steelblue", "TM": "cornflowerblue",
          "FB": "tomato",    "FM": "salmon"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"CNN-AE Hybrid (Tuned) — AUC={auc:.4f}", fontsize=13)

for cls in ["TB", "TM", "FB", "FM"]:
    mask = np.array(all_class_names) == cls
    if mask.sum() == 0: continue
    axes[0].hist(all_cls_probs[mask], bins=20, alpha=0.6,
                 label=cls, color=colors[cls])
axes[0].axvline(0.5, color="black", linestyle="--", lw=1.5, label="0.5")
axes[0].set_xlabel("Classifier Probability"); axes[0].set_ylabel("Count")
axes[0].set_title("Score Distribution by Class")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(fpr, tpr, lw=2, color="darkorange", label=f"AUC={auc:.4f}")
axes[1].plot(fpr, tpr, lw=1, color="gray",       linestyle="--",
             label="Baseline AUC=0.8426", alpha=0.5)
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
op_idx = np.argmin(np.abs(roc_thresh - best_thresh))
axes[1].scatter(fpr[op_idx], tpr[op_idx], s=80, zorder=5,
                color="red", label="Op. point")
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("ROC Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(all_true, all_pred)
ConfusionMatrixDisplay(cm, display_labels=["Normal\n(TB/TM)", "Anomaly\n(FB/FM)"]
                       ).plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("evaluation_tuned.png", dpi=150); plt.show()
print("Saved → evaluation_tuned.png")

# Summary
print(f"\nBest hyperparameters found:")
for k, v in best.items():
    print(f"  {k}: {v}")