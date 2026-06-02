# Gold-Standard Test Set: Reviewer Guide

In Week 3, Reviewer 1 and Reviewer 2 will independently rate 300 anonymized CXRs. This guide ensures consistency.

## Rating Protocol
1. **Tooling:** Use the local Streamlit RatingTool app.
2. **Environment:** Review images in a dimly lit room to maximize screen contrast.
3. **Task:** For each image, you will assign an integer (1, 2, or 3) for each of the 7 axes.
   * `1` = Acceptable
   * `2` = Borderline
   * `3` = Repeat

## Tie-Breaking & Ambiguity
* If an image borders between `Acceptable` and `Borderline`, default to **Acceptable** unless the flaw actively slows down your theoretical diagnostic process.
* If an image borders between `Borderline` and `Repeat`, default to **Repeat** if any major pathology could easily hide in the affected area (e.g., a small pneumothorax masked by rotation/blur).
* **Do not look at each other's ratings.** Independence is strictly required to calculate an accurate Cohen's kappa (inter-rater reliability).