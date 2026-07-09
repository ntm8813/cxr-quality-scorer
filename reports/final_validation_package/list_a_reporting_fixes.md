# List A Reporting Fixes — MTV-INT-RAD-003

This report joins existing kappa/confusion-matrix outputs from `compute_kappa.py` and `compute_validation.py`, and adds the honesty checks the post-delivery review asked for: kappa read against its human ceiling, minority-class catch rate, and an explicit explanation of the overall-vs-per-axis kappa gap.

## 1. Model kappa vs. human inter-rater ceiling

| Axis | Model κ | Human Ceiling κ | % of Ceiling | Honest line |
|---|---|---|---|---|
| sharpness | 0.0254 | 0.2605 | 9.8% | 0.0254 against a human ceiling of 0.2605 |
| exposure | 0.0032 | 0.5841 | 0.5% | 0.0032 against a human ceiling of 0.5841 |
| rotation | 0.0317 | 0.3019 | 10.5% | 0.0317 against a human ceiling of 0.3019 |
| coverage | 0.0604 | 0.328 | 18.4% | 0.0604 against a human ceiling of 0.3280 |
| inspiration | 0.0021 | -0.0494 | -4.3% | 0.0021 against a human ceiling of -0.0494 |
| artifact | 0.0443 | 0.453 | 9.8% | 0.0443 against a human ceiling of 0.4530 |
| metadata | 0.0 | -0.0071 | -0.0% | 0.0000 against a human ceiling of -0.0071 |
| OVERALL | 0.2205 | 0.5248 | 42.0% | 0.2205 against a human ceiling of 0.5248 |

## 2. Minority-class catch rate (borderline + repeat)

Agreement-% and kappa are dominated by the acceptable majority class. This is the number that actually answers "does the model catch bad images":

| Axis | Catch Rate | Actual Bad Cases | Caught |
|---|---|---|---|
| sharpness | 10.7% | 75 | 8 |
| exposure | 0.9% | 111 | 1 |
| rotation | 91.4% | 58 | 53 |
| coverage | 1.4% | 70 | 1 |
| inspiration | 100.0% | 109 | 109 |
| artifact | 90.3% | 144 | 130 |
| metadata | 0.0% | 5 | 0 |
| OVERALL | 51.8% | 112 | 58 |

## 3. Per-class precision / recall / F1

### sharpness

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | 0.7546 | 0.9156 | 0.8273 | 225 |
| borderline | 0.2963 | 0.1081 | 0.1584 | 74 |
| repeat | None | 0.0 | None | 1 |

### exposure

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | 0.6309 | 0.9947 | 0.7721 | 189 |
| borderline | 0.5 | 0.0096 | 0.0189 | 104 |
| repeat | None | 0.0 | None | 7 |

### rotation

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | 0.9138 | 0.219 | 0.3533 | 242 |
| borderline | 0.2 | 0.0862 | 0.1205 | 58 |
| repeat | 0.0 | None | None | 0 (support=0 — no real examples of this class in the gold-standard set, so precision/recall here are not meaningful regardless of the model's behaviour.) |

### coverage

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | 0.7661 | 0.9826 | 0.861 | 230 |
| borderline | 0.0 | 0.0 | None | 67 |
| repeat | 1.0 | 0.3333 | 0.5 | 3 |

### inspiration

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | None | 0.0 | None | 191 |
| borderline | 0.3607 | 0.6055 | 0.4521 | 109 |
| repeat | 0.0 | None | None | 0 (support=0 — no real examples of this class in the gold-standard set, so precision/recall here are not meaningful regardless of the model's behaviour.) |

### artifact

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | 0.6818 | 0.1923 | 0.3 | 156 |
| borderline | 0.5 | 0.0141 | 0.0274 | 142 |
| repeat | 0.0079 | 1.0 | 0.0157 | 2 |

### metadata

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | 0.9833 | 1.0 | 0.9916 | 295 |
| borderline | None | 0.0 | None | 5 |
| repeat | None | None | None | 0 (support=0 — no real examples of this class in the gold-standard set, so precision/recall here are not meaningful regardless of the model's behaviour.) |

### OVERALL

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| acceptable | 0.7017 | 0.6755 | 0.6883 | 188 |
| borderline | 0.3983 | 0.4896 | 0.4393 | 96 |
| repeat | 1.0 | 0.0625 | 0.1176 | 16 |

## 4. Why overall κ (0.22) doesn't match near-zero per-axis κ

- Overall kappa: **0.2205**
- Mean per-axis kappa: **0.0239**
- Max per-axis kappa: **0.0604**
- Gap: **0.1966**

Overall kappa is NOT an average or aggregate of the seven per-axis kappas. It is computed independently as kappa(model.overall_flag, reviewer.global_rating) — the model's fused composite-score flag against the reviewer's separate holistic global rating column. Per-axis kappa is computed as kappa(model.<axis>_flag, reviewer.<axis>) for each of the seven axes individually. These are two distinct computations on two distinct column pairs (see compute_overall_kappa() vs compute_per_axis_kappa() in src/analysis/compute_validation.py). There is no mathematical requirement that overall track the per-axis mean, and a higher overall than per-axis mean does not indicate a bug — it indicates the composite score's weighted-sum fusion (src/fusion/score_fusion.py) can land in approximately the right overall bucket via compensating errors across axes or by being dominated by whichever axis carries the most weight, even while no individual axis reliably tracks its own corresponding reviewer rating. This is precisely why per-axis kappa, not overall kappa alone, is the correct metric for axis-level model quality.
