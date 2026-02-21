import pydicom
import numpy as np
import pandas as pd
from pathlib import Path

class PreProcessData:
    def __init__(
        self,
        import_dir,
        export_dir,
        uuid_prefix: str = "",
    ):
        self.import_dir = Path(import_dir)
        self.export_dir = Path(export_dir)
        self.uuid_prefix = uuid_prefix

    def run(self):
        # Iterate through each folder in the import directory
        # Each folder is a CT scan comprised of a set of .dcm files
        for folder in self.import_dir.iterdir():
            if not folder.is_dir():
                continue

            # apply prefix so that identical UUIDs from different
            # experiments do not collide
            out_name = f"{self.uuid_prefix}{folder.name}"
            export_subdir = self.export_dir / out_name
            export_subdir.mkdir(parents=True, exist_ok=True)

            # Get all dcm metadata
            all_slices = list(folder.glob("*.dcm"))
            ct_data = []
            for dcm in all_slices:
                ds = pydicom.dcmread(dcm)
                ct_data.append(ds)

            # sort & convert to HU, stack, etc.
            ct_data_sorted = sorted(ct_data, key=lambda ds: float(ds.ImagePositionPatient[2]))

            # Iterate through each slice in order, get their hu value, and store in npy_slices
            npy_slices = []
            for ds in ct_data_sorted:
                pixel_array = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))

                hu = pixel_array * slope + intercept

                # Clip HU values to the range [-1000, 400] then scale to [0,1] range
                # Air ~ 0, Lung Tissue ~0.2-0.3, Soft Tissue ~ 0.7, Bone ~ 1.0
                hu = np.clip(hu, -1000, 400)
                slice_image = (hu + 1000) / 1400

                npy_slices.append(slice_image)

            # Save the full volume as a single .npz
            volume = np.stack(npy_slices, axis=0)
            np.savez(export_subdir / "volume.npz", volume=volume)

if __name__ == "__main__":
    exp1 = PreProcessData(
        import_dir="data/Experiment 1 - Blind",
        export_dir="processed_dataset",
        uuid_prefix="exp1_",
    )
    exp1.run()

    exp2 = PreProcessData(
        import_dir="data/Experiment 2 - Blind",
        export_dir="processed_dataset",
        uuid_prefix="exp2_",
    )
    exp2.run()