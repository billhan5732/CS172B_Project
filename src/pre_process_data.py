import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from config import *

class PreProcessData:
    def __init__(self, importDirectory, importLabels, exportDirectory):
        self.import_dir = Path(importDirectory)
        self.export_dir = Path(exportDirectory)

        # read CSV
        self.df = pd.read_csv(importLabels)
        print(f"Initial Labels size: {len(self.df)}")

        # Optionally debug / clean
        if DEBUG:
            self.debug_clean_labels()
            print(f"Final Labels size: {len(self.df)}")

        # after cleaning, create arrays
        self.filenames = self.df['uuid'].to_numpy()

        self.labels = self.df['type'].to_numpy()

    def debug_clean_labels(self):

        folder_uuids = {folder.name.strip().lower() for folder in self.import_dir.iterdir() if folder.is_dir()}
        folder_set = set(folder_uuids)

        filtered_rows = []

        for idx, row in self.df.iterrows():
            if str(row['uuid']) in folder_set:
                #print(f"UUID Matched: {row['uuid']}")
                filtered_rows.append(row)

        df_filtered = pd.DataFrame(filtered_rows).reset_index(drop=True)

        self.df = df_filtered

    def run(self):
        for slicesFolder in self.import_dir.iterdir():
            if not slicesFolder.is_dir():
                continue

            export_subdir = self.export_dir / slicesFolder.name
            export_subdir.mkdir(parents=True, exist_ok=True)

            if EXPORT_NPZ:
                npy_slices = []

            for slice in slicesFolder.iterdir():
                if slice.is_dir() or (slice.suffix.lower() != ".dcm"):
                    continue

                dcm = pydicom.dcmread(slice)
                pixel_array = dcm.pixel_array.astype(np.float32)

                slope = float(getattr(dcm, "RescaleSlope", 1.0))
                intercept = float(getattr(dcm, "RescaleIntercept", 0.0))

                hu = pixel_array * slope + intercept

                hu = np.clip(hu, -1000, 400)
                slice_image = (hu + 1000) / 1400
                # print(f"Max: {pixel_array.max()}") # -2048
                # print(f"Min: {pixel_array.min()}") # 2048

                # print(pixel_array.shape) # 512x512
                # print(pixel_array.dtype) # int16

                if not EXPORT_NPZ:
                    # Save each slice as .npy
                    output_path = export_subdir / (slice.stem + ".npy")
                    np.save(output_path, slice_image)
                else:
                    # Collect slices for later saving
                    npy_slices.append(slice_image)

            # After the loop, save the full volume as a single .npz
            if EXPORT_NPZ:
                volume_array = np.stack(npy_slices, axis=0)  # shape: (num_slices, H, W)
                output_path = export_subdir / "volume.npz"
                np.savez(output_path, volume=volume_array)


app = PreProcessData("data/Experiment 1 - Blind","data/labels_exp1.csv","data/export")
app.run()