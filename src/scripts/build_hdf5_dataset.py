import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.io.dicom_reader import DICOMReader
import uuid

def build_dataset(raw_dir: str, hdf5_path: str, manifest_path: str, max_images: int = 1500):
    reader = DICOMReader()
    
    # Get list of PNGs from the raw directory
    valid_files = [f for f in os.listdir(raw_dir) if f.endswith('.png')]
    if len(valid_files) > max_images:
        valid_files = valid_files[:max_images]
        
    print(f"Found {len(valid_files)} images. Starting preprocessing...")
    
    manifest_data = []
    
    with h5py.File(hdf5_path, 'w') as h5f:
        for filename in tqdm(valid_files, desc="Processing CXRs"):
            filepath = os.path.join(raw_dir, filename)
            try:
                # Using load_from_png since NIH data is in PNG format
                image, metadata = reader.load_from_png(filepath)
                
                # Generate a mock StudyInstanceUID for NIH PNGs if not present
                study_uid = metadata.get("study_uid") or str(uuid.uuid4())
                metadata["study_uid"] = study_uid
                
                # Save to HDF5 keyed by StudyInstanceUID
                h5f.create_dataset(study_uid, data=image, compression="gzip")
                
                # Append to manifest
                manifest_data.append({
                    "study_uid": study_uid,
                    "original_filename": filename,
                    "split": "train" # Will be adjusted later if needed
                })
                
            except Exception as e:
                print(f"\nFailed to process {filename}: {str(e)}")
                
    # Save manifest
    df = pd.DataFrame(manifest_data)
    df.to_csv(manifest_path, index=False)
    print(f"\nSuccessfully wrote HDF5 to {hdf5_path}")
    print(f"Successfully wrote manifest to {manifest_path}")

if __name__ == "__main__":
    # Define paths
    RAW_DIR = "data/raw/nih_subset"
    PROCESSED_DIR = "data/processed"
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    HDF5_OUTPUT = os.path.join(PROCESSED_DIR, "cxr_clean.h5")
    MANIFEST_OUTPUT = os.path.join(PROCESSED_DIR, "manifest.csv")
    
    build_dataset(RAW_DIR, HDF5_OUTPUT, MANIFEST_OUTPUT)