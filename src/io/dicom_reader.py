import numpy as np
import cv2
import pydicom
from PIL import Image


class DICOMReader:

    def load(self, path: str):

        # Case 1: DICOM
        if path.lower().endswith(".dcm"):
            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array.astype(np.float32)

        # Case 2: PNG/JPG
        else:
            img = np.array(Image.open(path).convert("L"), dtype=np.float32)

        img = img / 255.0

        metadata = self._build_metadata(path)

        return img, metadata

    # REQUIRED BY TESTS (fix #1)
    def load_from_png(self, path: str):
        img, metadata = self.load(path)
        return img, metadata

    def _build_metadata(self, path: str):
        filename = path.split("\\")[-1].split("/")[-1]

        return {
            "study_uid": filename,
            "modality": "CR",
            "view_position": "PA",
            "body_part": "CHEST",
            "exposure_index": 0.0,
            "deviation_index": 0.0,
            "kvp": 120,
            "patient_id": "UNKNOWN",
            "source_path": path
        }