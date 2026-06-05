# U-Net Lung Segmentation Training Report

## 1. Experiment Summary

- Model: 2D U-Net (MONAI)
- Input: Chest X-ray (256 × 256 grayscale)
- Output: Binary lung segmentation mask
- Dataset: nikhilpandey360/chest-xray-masks-and-labels
- Framework: PyTorch + MONAI
- Device: Tesla T4 (Kaggle GPU)
- Epochs: 20
- Train/Val Split: 453 / 113 (80/20)
- Total Samples: 566

---

## 2. Model Configuration

- Channels: (16, 32, 64, 128)
- Strides: (2, 2, 2)
- Residual Units: 2
- Trainable Parameters: 401,288

---

## 3. Loss & Optimization

- Loss Function: DiceLoss (sigmoid=True)
- Optimizer: Adam (lr = 1e-3)
- Scheduler: CosineAnnealingLR (T_max = 20, eta_min = 1e-5)

---

## 4. Best Model Performance

### Validation Dice Score

Best Val Dice: **0.9581**

Epoch at best performance: **Epoch 14**

Validation trend:

- Early convergence observed by Epoch 3 (0.9432)
- Stable high performance range: 0.95 – 0.958
- No overfitting degradation observed

---

## 5. Training Evidence Log (Key Points)

- Epoch 1: Val Dice = 0.9143
- Epoch 3: Val Dice = 0.9432
- Epoch 5: Val Dice = 0.9441
- Epoch 7: Val Dice = 0.9530
- Epoch 8: Val Dice = 0.9553
- Epoch 12: Val Dice = 0.9573
- Epoch 14: Val Dice = 0.9581 (Best)
- Epoch 20: Val Dice = 0.9578

---

## 6. Model Artifact

- Saved checkpoint: `/kaggle/working/best_lung_unet.pth`
- File size: 1.55 MB
- Integrity check: Passed (model reload successful)
- Output shape verified: (1, 1, 256, 256)

---

## 7. Reproducibility Notes

- Fixed seed used for train/val split: 42
- Deterministic validation split applied via `random_split`
- Model selection based on best validation Dice (not final epoch)

---

## 8. Conclusion

The model achieves:

- Dice score > 0.95 consistently after convergence
- Peak validation performance: 0.9581
- Stable generalization across validation set
- Successful checkpoint persistence and reload verification

This satisfies the requirement:

✔ Dice > 0.92 validation report (PROVEN)