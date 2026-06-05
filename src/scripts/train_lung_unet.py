from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.io.dicom_reader import DICOMReader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gray_image(path: str, size: int) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".dcm"):
        image, _ = DICOMReader().load(path)
        image = image.astype(np.float32)
        if image.shape != (size, size):
            image = np.asarray(
                Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
                    (size, size),
                    resample=Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS,
                ),
                dtype=np.float32,
            ) / 255.0
        return np.clip(image, 0.0, 1.0)

    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize(
            (size, size),
            resample=Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_size: int = 1024):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        valid_suffixes = {".png", ".jpg", ".jpeg", ".dcm"}
        images = [
            p for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ]

        pairs = []
        for img_path in images:
            stem = img_path.stem
            candidates = [
                self.mask_dir / f"{stem}.png",
                self.mask_dir / f"{stem}.jpg",
                self.mask_dir / f"{stem}.jpeg",
                self.mask_dir / f"{stem}.dcm",
            ]
            mask_path = next((p for p in candidates if p.exists()), None)
            if mask_path is not None:
                pairs.append((img_path, mask_path))

        if not pairs:
            raise ValueError("No paired image/mask files found")

        self.pairs = sorted(pairs, key=lambda x: x[0].name)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        image = load_gray_image(str(image_path), self.image_size)
        mask = load_gray_image(str(mask_path), self.image_size)

        image_t = torch.from_numpy(image[None, ...].astype(np.float32))
        mask_t = torch.from_numpy((mask > 0.5).astype(np.float32)[None, ...])

        return image_t, mask_t


def build_model() -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()

    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = ((2.0 * intersection + eps) / (denominator + eps)).mean().item()
    return float(dice)


def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device) -> float:
    model.train()
    running = 0.0
    total = 0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running += float(loss.item()) * batch_size
        total += batch_size

    return running / max(1, total)


@torch.no_grad()
def validate(model, loader, bce_loss, dice_loss, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    total = 0

    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
        dice = dice_score_from_logits(logits, masks)

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        running_dice += float(dice) * batch_size
        total += batch_size

    return running_loss / max(1, total), running_dice / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lung segmentation UNet.")
    parser.add_argument("--images", required=True, help="Directory with CXR images")
    parser.add_argument("--masks", required=True, help="Directory with binary lung masks")
    parser.add_argument("--output", default="weights/best_lung_unet.pth", help="Output checkpoint path")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = LungSegmentationDataset(args.images, args.masks, image_size=args.image_size)
    val_len = max(1, int(round(len(dataset) * args.val_split)))
    train_len = max(1, len(dataset) - val_len)

    if train_len + val_len > len(dataset):
        val_len = len(dataset) - train_len

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device(args.device)
    model = build_model().to(device)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(sigmoid=True, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)

    best_dice = -1.0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device)
        val_loss, val_dice = validate(model, val_loader, bce_loss, dice_loss, device)
        scheduler.step(val_dice)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.output)
            print(f"saved={args.output} best_dice={best_dice:.4f}")

    print(f"finished best_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()