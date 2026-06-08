from pathlib import Path
import random
import shutil
import pandas as pd

SOURCE_DIR = Path("data/raw/nih_subset")
TARGET_DIR = Path("data/gold_standard")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

all_images = sorted(SOURCE_DIR.rglob("*.png"))

print(f"Found {len(all_images)} images")

random.seed(42)
selected = random.sample(all_images, 300)

manifest_rows = []

for img in selected:
    dest = TARGET_DIR / img.name
    shutil.copy2(img, dest)

    manifest_rows.append({
        "study_uid": img.stem,
        "filename": img.name
    })

manifest = pd.DataFrame(manifest_rows)
manifest = manifest.sort_values("filename")

Path("manifests").mkdir(exist_ok=True)

manifest.to_csv(
    "manifests/gold_standard_manifest.csv",
    index=False
)

print(f"Copied {len(manifest)} images")
print("Saved manifest → data/gold_standard_manifest.csv")