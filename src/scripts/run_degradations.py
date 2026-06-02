import os
import h5py
import pandas as pd
from tqdm import tqdm
from src.data.degraders.degrade_blur import apply_blur
from src.data.degraders.degrade_noise import apply_noise
from src.data.degraders.degrade_exposure import apply_exposure
from src.data.degraders.degrade_rotation import apply_rotation

def orchestrate():
    CLEAN_HDF5 = "data/processed/cxr_clean.h5"
    OUT_HDF5 = "data/processed/cxr_degraded.h5"
    OUT_CSV = "data/processed/degradation_manifest.csv"
    
    if not os.path.exists(CLEAN_HDF5):
        raise FileNotFoundError(f"Missing {CLEAN_HDF5}. Did you run dvc pull?")
        
    manifest_rows = []
    
    transforms = {
        "blur": apply_blur,
        "noise": apply_noise,
        "exposure": apply_exposure,
        "rotation": apply_rotation
    }
    
    with h5py.File(CLEAN_HDF5, 'r') as h5_clean, h5py.File(OUT_HDF5, 'w') as h5_out:
        base_uids = list(h5_clean.keys())
        print(f"Loaded {len(base_uids)} clean images. Starting batch degradation pipeline...")
        
        for base_uid in tqdm(base_uids):
            clean_img = h5_clean[base_uid][:]
            
            # 1. Save the clean baseline (Severity 0)
            clean_new_uid = f"{base_uid}_clean_0"
            h5_out.create_dataset(clean_new_uid, data=clean_img, compression="gzip")
            manifest_rows.append({"uid": clean_new_uid, "base_uid": base_uid, "axis": "none", "severity": 0})
            
            # 2. Loop through all 4 axes and severities (1 and 2)
            for axis_name, func in transforms.items():
                for severity in [1, 2]:
                    try:
                        degraded_img = func(clean_img, severity)
                        new_uid = f"{base_uid}_{axis_name}_{severity}"
                        
                        h5_out.create_dataset(new_uid, data=degraded_img, compression="gzip")
                        manifest_rows.append({"uid": new_uid, "base_uid": base_uid, "axis": axis_name, "severity": severity})
                    except Exception as e:
                        print(f"Error modifying {base_uid} on {axis_name}: {e}")

    # Write data manifest log
    df = pd.DataFrame(manifest_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n🎉 Process Complete! Generated {len(df)} images.")
    print(f"Saved database to: {OUT_HDF5}")
    print(f"Saved layout log to: {OUT_CSV}")

if __name__ == "__main__":
    orchestrate()