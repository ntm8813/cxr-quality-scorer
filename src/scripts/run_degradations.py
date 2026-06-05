import os
import h5py
import pandas as pd
from tqdm import tqdm

from src.data.degraders.degrade_blur        import apply_blur
from src.data.degraders.degrade_noise       import apply_noise
from src.data.degraders.degrade_exposure    import apply_exposure
from src.data.degraders.degrade_rotation    import apply_rotation, get_applied_angle
from src.data.degraders.degrade_coverage    import apply_coverage
from src.data.degraders.degrade_inspiration import apply_inspiration


def orchestrate():
    CLEAN_HDF5 = "data/processed/cxr_clean.h5"
    OUT_HDF5   = "data/processed/cxr_degraded.h5"
    OUT_CSV    = "data/processed/degradation_manifest.csv"

    if not os.path.exists(CLEAN_HDF5):
        raise FileNotFoundError(f"Missing {CLEAN_HDF5}. Did you run dvc pull?")

    manifest_rows = []

    # Axes that accept base_uid for deterministic behaviour
    uid_aware = {"rotation", "exposure"}

    transforms = {
        "blur":        apply_blur,
        "noise":       apply_noise,
        "exposure":    apply_exposure,
        "rotation":    apply_rotation,
        "coverage":    apply_coverage,
        "inspiration": apply_inspiration,
    }

    with h5py.File(CLEAN_HDF5, "r") as h5_clean, \
         h5py.File(OUT_HDF5,   "w") as h5_out:

        base_uids = list(h5_clean.keys())
        print(f"Loaded {len(base_uids)} clean images. Starting degradation pipeline...")

        for base_uid in tqdm(base_uids):
            clean_img = h5_clean[base_uid][:]

            # Clean baseline
            clean_uid = f"{base_uid}_clean_0"
            ds = h5_out.create_dataset(clean_uid, data=clean_img, compression="gzip")
            ds.attrs["applied_angle_deg"] = 0.0  # ground truth for rotation eval
            manifest_rows.append({
                "uid": clean_uid, "base_uid": base_uid,
                "axis": "none", "severity": 0,
            })

            for axis_name, func in transforms.items():
                for severity in [1, 2]:
                    try:
                        if axis_name in uid_aware:
                            degraded_img = func(clean_img, severity, base_uid=base_uid)
                        else:
                            degraded_img = func(clean_img, severity)

                        new_uid = f"{base_uid}_{axis_name}_{severity}"
                        ds = h5_out.create_dataset(
                            new_uid, data=degraded_img, compression="gzip"
                        )

                        # Store ground-truth angle so evaluation never has to
                        # re-estimate it from the image.
                        if axis_name == "rotation":
                            ds.attrs["applied_angle_deg"] = get_applied_angle(
                                severity, base_uid
                            )

                        manifest_rows.append({
                            "uid": new_uid, "base_uid": base_uid,
                            "axis": axis_name, "severity": severity,
                        })
                    except Exception as exc:
                        print(f"  Error on {base_uid}/{axis_name} sev{severity}: {exc}")

    df = pd.DataFrame(manifest_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nDone. {len(df)} entries written.")
    print(f"  HDF5 -> {OUT_HDF5}")
    print(f"  CSV  -> {OUT_CSV}")


if __name__ == "__main__":
    orchestrate()