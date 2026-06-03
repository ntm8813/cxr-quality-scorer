import os
import numpy as np
import torch
from src.ml.model_registry import ModelRegistry
from src.io.dicom_reader import DICOMReader


def to_tensor(img):
    t = torch.tensor(img, dtype=torch.float32)
    if len(t.shape) == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    return t


def dice(mask1, mask2, eps=1e-6):
    mask1 = mask1.flatten()
    mask2 = mask2.flatten()
    intersection = (mask1 * mask2).sum()
    return (2.0 * intersection + eps) / (mask1.sum() + mask2.sum() + eps)


def main():
    data_dir = r"C:\Users\nirma\Documents\cxr-quality-scorer\data\raw\nih_subset"
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".dcm"))
    ][:100]

    reader = DICOMReader()
    registry = ModelRegistry()
    model = registry.load_lung_segmentation()
    model.eval()

    dice_scores = []

    for path in files:
        img, meta = reader.load(path)
        img_t = to_tensor(img)

        with torch.no_grad():
            pred1 = model(img_t)
            pred2 = model(img_t + torch.randn_like(img_t) * 0.01)

        pred1 = (pred1 > 0.5).float()
        pred2 = (pred2 > 0.5).float()

        score = dice(pred1.cpu().numpy(), pred2.cpu().numpy())
        dice_scores.append(score)

    mean_dice = float(np.mean(dice_scores))

    print("\nCXR SEGMENTATION ROBUSTNESS REPORT")
    print(f"\nSamples evaluated : {len(dice_scores)}")
    print(f"Mean Dice score   : {mean_dice:.4f}")

    if mean_dice > 0.92:
        print("\nRequirement (>0.92): PASS")
    else:
        print("\nRequirement (>0.92): FAIL")

    os.makedirs("reports", exist_ok=True)
    with open("reports/dice_report.txt", "w", encoding="utf-8") as f:
        f.write(
            f"CXR SEGMENTATION ROBUSTNESS REPORT\n\n"
            f"Samples evaluated : {len(dice_scores)}\n"
            f"Mean Dice score   : {mean_dice:.4f}\n"
        )


if __name__ == "__main__":
    main()