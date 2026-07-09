from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


class DICOMReader:
    def load(self, path: str):
        if path.lower().endswith(".dcm"):
            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array.astype(np.float32)
            metadata = self._build_metadata(path, ds)
        else:
            img = np.array(Image.open(path).convert("L"), dtype=np.float32)
            metadata = self._build_metadata(path, None)

        img = img / 255.0
        return img, metadata

    # REQUIRED BY TESTS (fix #1)
    def load_from_png(self, path: str):
        return self.load(path)

    def _build_metadata(self, path: str, ds=None):
        filename = Path(path).name

        if ds is None:
            return {
                "study_uid": filename,
                "modality": "UNKNOWN",
                "view_position": "UNKNOWN",
                "body_part": "UNKNOWN",
                "exposure_index": 0.0,
                "deviation_index": 0.0,
                "kvp": 0.0,
                "patient_id": "UNKNOWN",
                "source_path": path,
            }

        return {
            "study_uid": filename,
            "modality": self._text(ds, "Modality"),
            "view_position": self._first_text(ds, ("ViewPosition", "PatientPosition")),
            "body_part": self._text(ds, "BodyPartExamined"),
            "exposure_index": self._number(ds, "ExposureIndex", 0.0),
            "deviation_index": self._number(ds, "DeviationIndex", 0.0),
            "kvp": self._number(ds, "KVP", 0.0),
            "patient_id": self._text(ds, "PatientID"),
            "source_path": path,
        }

    @staticmethod
    def _unwrap(value):
        if hasattr(value, "value"):
            value = value.value
        return value

    @classmethod
    def _text(cls, ds, key, default="UNKNOWN"):
        try:
            value = cls._unwrap(ds.get(key, default))
        except Exception:
            return default

        if value is None:
            return default

        if hasattr(value, "CodeMeaning"):
            value = value.CodeMeaning

        if isinstance(value, (list, tuple)):
            for item in value:
                item = cls._unwrap(item)
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    return text
            return default

        text = str(value).strip()
        return text or default

    @classmethod
    def _first_text(cls, ds, keys, default="UNKNOWN"):
        for key in keys:
            text = cls._text(ds, key, default=None)
            if text:
                return text
        return default

    @classmethod
    def _number(cls, ds, key, default=0.0):
        try:
            value = cls._unwrap(ds.get(key, default))
        except Exception:
            return default

        if value is None:
            return default

        if isinstance(value, (list, tuple)):
            value = value[0] if value else None

        try:
            return float(value)
        except (TypeError, ValueError):
            return default