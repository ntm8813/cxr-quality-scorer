# Week 4 Status Report — MTV-INT-RAD-003

## Automated CXR Image Quality Scorer — Reviewer Validation

---

## 1. Summary

Week 4 validated the full pipeline against a 300-image gold-standard test set rated
by two independent reviewers. During the initial validation pass, a systematic
investigation uncovered and resolved three critical bugs, improving the overall
weighted Cohen's κ against reviewer consensus from **0.0496 to 0.2205** — a 4.4×
improvement — and agreement from **43.7% to 58.3%**.

---

## 2. Initial Validation Pass (Pre-Bugfix)

The first end-to-end validation run against the 300-image gold standard produced:

- Overall weighted κ: **0.0496** (near chance-level agreement)
- Overall agreement: **43.7%**
- Composite score distribution collapsed toward the "repeat" end of the scale
- Rotation axis: **280/300 studies (93%)** flagged as severe rotation, while
  reviewer consensus showed **0 studies** rated as requiring repeat due to rotation

This result triggered a root-cause investigation rather than parameter tuning,
since a 93% false-positive rate on one axis was implausible given the visual
quality of the source images.

---

## 3. Bug Discovery and Resolution

### Bug 1 — Silent checkpoint loading failure (critical)

`ModelRegistry.load_lung_segmentation()` used
`model.load_state_dict(checkpoint, strict=False)`.

Direct key-set inspection revealed the saved checkpoint's keys had no prefix
(e.g. `0.conv.unit0.adn.A.weight`), while the live MONAI `UNet` wrapper expected
a `model.` prefix (e.g. `model.0.conv.unit0.adn.A.weight`). The existing
`_sanitize_state_dict()` only stripped known prefixes (`module.`) and did nothing
when the checkpoint had no prefix at all to strip.

**Result: `strict=False` silently loaded 0 of 49 keys.** The segmentation network
ran at random initialization for the entire initial validation pass with no error
or warning surfaced anywhere in the pipeline.

Verified directly: