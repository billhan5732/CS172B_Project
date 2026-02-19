import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

VALID_CLASSES = {"FB", "FM", "TB", "TM"}
CLASS_TO_IDX = {"FB": 0, "FM": 1, "TB": 2, "TM": 3}
IDX_TO_CLASS = {0: "FB", 1: "FM", 2: "TB", 3: "TM"}

# NOTE: Works for 2D CNN, maybe need to change around later to work with other approaches?
# Custom dataset for extracting patches from CT scans
class CTPatchDataset(Dataset):
    def __init__(
        self, 
        export_root: Path,
        labels_csv: Path,
        patch_size: int,
        augment: bool,
        jitter_px: int
        ):

        super().__init__()

        self.export_root = Path(export_root)
        self.labels_csv = Path(labels_csv)
        self.patch_size = patch_size
        self.augment = augment
        self.jitter_px = jitter_px

        self._validate_config()

        self.df = pd.read_csv(self.labels_csv)

        # Cache so we don't constantly reload volume.npz
        self._cache = dict()

    def __len__(self) -> int:
        return len(self.df)

    # Gets a single item (slice) from labels csv at the given idx
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the row from the dataframe at idx
        row = self.df.iloc[idx]

        # Get information from row
        label_type = str(row["type"])
        uuid = str(row["uuid"])
        scan_slice = int(row["slice"])
        x = int(row["x"])
        y = int(row["y"])

        # Validate label type
        if label_type not in CLASS_TO_IDX:
            raise ValueError(f"Invalid label type '{label_type}' at idx={idx}")

        # Build path and get volume
        volume_path = self.export_root / uuid / "volume.npz"

        if uuid not in self._cache:
            self._cache[uuid] = np.load(volume_path, allow_pickle=False)["volume"] # Idk what allow_pickles does but gpt said to do it

        volume = self._cache[uuid]

        # Validate scan_slice in bounds
        if not (0 <= scan_slice < volume.shape[0]):
            raise IndexError(
                f"slice={scan_slice} out of bounds for uuid={uuid} (Z={volume.shape[0]})"
            )

        # Get the specific slice of interest
        img = volume[scan_slice]

        # NOTE: Apply random jittering to the center coordinates during training
        if self.augment and self.jitter_px > 0:
            dx = random.randint(-self.jitter_px, self.jitter_px)
            dy = random.randint(-self.jitter_px, self.jitter_px)
            x += dx
            y += dy
            
            # Just incase jitter makes our (x,y) go out of bounds
            x = max(0, min(x, img.shape[1] - 1))
            y = max(0, min(y, img.shape[0] - 1))

        # Crop the patch from the image
        patch = self.crop_center(img, x, y)

        # Need to convert to torch tensors to feed into CNN
        patch = torch.from_numpy(patch).float().unsqueeze(0) # (1, patch_size, patch_size)
        label = torch.tensor(CLASS_TO_IDX[label_type], dtype=torch.long)

        return (patch, label)

    def _validate_config(self):
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")

        if not self.export_root.exists():
            raise FileNotFoundError(f"Export root not found: {self.export_root}")

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")

        if self.jitter_px < 0:
            raise ValueError("jitter_px must be >= 0")
    
    def crop_center(self, img: np.ndarray, x: int, y: int) -> np.ndarray:
        # Verify shape
        if img.ndim != 2:
            raise ValueError("img must be 2D")
        
        # This should never occur but I'll add it anyways why not
        # Check bounds
        H, W = img.shape
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"(x,y)=({x},{y}) out of bounds for image shape {(H, W)}")
        
        half_patch = self.patch_size // 2

        # Padding added to ensure we don't go out of bounds (i.e. if (x,y) is near edge of image)
        padded = np.pad(img, pad_width=half_patch, mode='constant', constant_values=0)

        x_pad = x + half_patch
        y_pad = y + half_patch

        # NOTE: +1 added to odd patch_size upper_bound to ensure patch is true to patch_size
        if self.patch_size % 2 == 1:
            patch = padded[
                y_pad-half_patch : y_pad+half_patch+1,
                x_pad-half_patch : x_pad+half_patch+1
            ]
        else:
            patch = padded[
                y_pad-half_patch : y_pad+half_patch,
                x_pad-half_patch : x_pad+half_patch
            ]

        if patch.shape != (self.patch_size, self.patch_size):
            raise ValueError(f"Unexpected patch shape: {patch.shape}")

        return patch
