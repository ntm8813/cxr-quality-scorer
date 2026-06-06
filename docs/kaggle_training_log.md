# Kaggle Training Log

## Day 15 — Blur Classifier

### Objective

Train a CNN-based blur detector for chest X-ray quality assessment.

### Dataset

NIH Chest X-ray dataset:

* 400 source images
* Resized to 224×224
* Grayscale replicated to 3 channels

### Synthetic Blur Generation

Each source image produced:

* 1 clean sample
* 3 Gaussian-blurred samples

Blur sigmas:

* 2.5
* 5.0
* 9.0

Final dataset:

* Clean: 400
* Blurred: 1200
* Total: 1600

### Model

* EfficientNet-B0
* Pretrained ImageNet weights
* Binary classifier
* BCEWithLogitsLoss
* Adam optimizer
* CosineAnnealingLR scheduler

### Result

* Best Validation AUC: 1.0000
* Reload Check: PASSED

### Artifact Produced

best_blur_classifier.pth

### Note

The model was trained on synthetic Gaussian blur and has not yet been validated on real patient-motion blur.

---

## Day 16 — Artifact Classifier

### Objective

Train a CNN-based artifact detector for chest X-ray quality assessment.

### Dataset

NIH Chest X-ray dataset:

* 400 source images
* Resized to 224×224
* Grayscale replicated to 3 channels

### Synthetic Artifact Generation

Each source image produced:

* 1 clean sample
* 1 grid-line artifact sample
* 1 foreign-object artifact sample
* 1 processing-halo artifact sample

Final dataset:

* Clean: 400
* Artifact: 1200
* Total: 1600

### Model

* EfficientNet-B0
* Pretrained ImageNet weights
* Binary classifier
* BCEWithLogitsLoss
* Adam optimizer
* CosineAnnealingLR scheduler

### Result

* Best Validation AUC: 1.0000
* Reload Check: PASSED

### Artifact Produced

best_artifact_classifier.pth

### Note

The model was trained on synthetic artifacts and has not yet been validated on real clinical artifacts.

---

## Summary

Completed CNN prototypes:

1. Blur detection model
2. Artifact detection model

Both models achieved:

* Validation AUC = 1.0000
* Successful checkpoint reload verification

These models serve as initial learned quality-assessment components for future integration into the chest X-ray quality scoring pipeline.
