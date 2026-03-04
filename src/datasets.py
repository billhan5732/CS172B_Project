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

class CTPatchDataset25D(Dataset):
    """
    2.5D dataset: returns a K-slice stack as channels.

    Output:
      x: (K, patch_size, patch_size) float32
      y: torch.long (binary if binary=True, else 4-class)
    """

    def __init__(
        self,
        export_root: Path,
        labels_csv: Path,
        patch_size: int,
        k_slices: int,
        pad_mode: str,          # "edge" or "constant"
        augment: bool,
        jitter_px: int,
        volume_key: str = "volume",
        binary: bool = False,   # if True, uses row["target"] (0/1) instead of 4-class "type"
    ):
        super().__init__()

        if k_slices <= 0 or k_slices % 2 == 0:
            raise ValueError("k_slices must be a positive odd integer (e.g., 3,5,7)")
        if pad_mode not in {"edge", "constant"}:
            raise ValueError("pad_mode must be 'edge' or 'constant'")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if jitter_px < 0:
            raise ValueError("jitter_px must be >= 0")

        self.export_root = Path(export_root)
        self.labels_csv = Path(labels_csv)
        self.patch_size = int(patch_size)
        self.k_slices = int(k_slices)
        self.pad_mode = pad_mode
        self.augment = bool(augment)
        self.jitter_px = int(jitter_px)
        self.volume_key = volume_key
        self.binary = bool(binary)

        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")
        if not self.export_root.exists():
            raise FileNotFoundError(f"Export root not found: {self.export_root}")

        self.df = pd.read_csv(self.labels_csv)

        # IMPORTANT: your layout requires source + uuid
        required_cols = {"uuid", "source", "slice", "x", "y"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"labels_csv is missing columns required by your folder layout: {sorted(missing)}"
            )

        if self.binary and "target" not in self.df.columns:
            raise ValueError("binary=True requires a 'target' column in the CSV")

        if (not self.binary) and "type" not in self.df.columns:
            raise ValueError("binary=False requires a 'type' column in the CSV")

        self._cache: Dict[str, np.ndarray] = {}  # cache by volume_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        source = str(row["source"])
        uuid = str(row["uuid"])
        volume_id = f"{source}_{uuid}"

        z_center = int(row["slice"])
        x = int(row["x"])
        y = int(row["y"])

        # Label
        if self.binary:
            label = torch.tensor(int(row["target"]), dtype=torch.long)  # 0/1
        else:
            label_type = str(row["type"])
            if label_type not in CLASS_TO_IDX:
                raise ValueError(f"Invalid label type '{label_type}' at idx={idx}")
            label = torch.tensor(CLASS_TO_IDX[label_type], dtype=torch.long)

        # Your volume path: processed_dataset/<source>_<uuid>/volume.npz
        volume_path = self.export_root / volume_id / "volume.npz"

        if volume_id not in self._cache:
            if not volume_path.exists():
                raise FileNotFoundError(f"Missing volume: {volume_path}")
            self._cache[volume_id] = np.load(
                volume_path, mmap_mode="r", allow_pickle=False
            )[self.volume_key]

        volume = self._cache[volume_id]  # (Z,H,W)
        Z, H, W = volume.shape

        if not (0 <= z_center < Z):
            raise IndexError(f"slice={z_center} out of bounds for volume_id={volume_id} (Z={Z})")

        # Augment: jitter x,y (same x,y applied across all K slices)
        if self.augment and self.jitter_px > 0:
            x += random.randint(-self.jitter_px, self.jitter_px)
            y += random.randint(-self.jitter_px, self.jitter_px)
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))

        # Build z indices for 2.5D stack
        half_k = self.k_slices // 2
        if self.pad_mode == "edge":
            z_indices = [max(0, min(Z - 1, z_center + dz)) for dz in range(-half_k, half_k + 1)]
            imgs = [volume[z] for z in z_indices]
        else:
            imgs = []
            for dz in range(-half_k, half_k + 1):
                z = z_center + dz
                if 0 <= z < Z:
                    imgs.append(volume[z])
                else:
                    imgs.append(np.zeros((H, W), dtype=volume.dtype))
        
        # Crop patches from each slice and stack as channels
        patches = [self.crop_center(img, x, y) for img in imgs]   # list of (P,P)
        stack = np.stack(patches, axis=0).astype(np.float32)      # (K,P,P)


        x_tensor = torch.from_numpy(stack)  # (K,P,P)
        return x_tensor, label

    def crop_center(self, img: np.ndarray, x: int, y: int) -> np.ndarray:
        if img.ndim != 2:
            raise ValueError("img must be 2D")

        H, W = img.shape
        if not (0 <= x < W and 0 <= y < H):
            raise ValueError(f"(x,y)=({x},{y}) out of bounds for image shape {(H, W)}")

        half_patch = self.patch_size // 2

        # Edge padding avoids artificial black borders
        padded = np.pad(img, pad_width=half_patch, mode="edge")

        x_pad = x + half_patch
        y_pad = y + half_patch

        if self.patch_size % 2 == 1:
            patch = padded[
                y_pad - half_patch : y_pad + half_patch + 1,
                x_pad - half_patch : x_pad + half_patch + 1,
            ]
        else:
            patch = padded[
                y_pad - half_patch : y_pad + half_patch,
                x_pad - half_patch : x_pad + half_patch,
            ]

        if patch.shape != (self.patch_size, self.patch_size):
            raise ValueError(f"Unexpected patch shape: {patch.shape}")

        return patch