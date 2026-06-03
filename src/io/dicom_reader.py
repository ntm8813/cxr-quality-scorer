import numpy as np
import cv2
import pydicom
from PIL import Image


class DICOMReader:

    def load(self, path: str):

        # Case 1: real DICOM
        if path.lower().endswith(".dcm"):
            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array.astype(np.float32)

        # Case 2: PNG/JPG dataset (YOUR CURRENT CASE)
        else:
            img = np.array(Image.open(path).convert("L"), dtype=np.float32)

        # Normalize
        img = img / 255.0

        metadata = {
            "study_uid": path.split("\\")[-1],
            "source_path": path
        }

        return img, metadata