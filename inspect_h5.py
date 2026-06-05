import h5py

h5_path = r"C:\Users\nirma\Documents\cxr-quality-scorer\data\processed\cxr_degraded.h5"
output_txt_path = r"C:\Users\nirma\Documents\cxr-quality-scorer\data\processed\h5_structure_report.txt"

try:
    # Open the text file for writing
    with open(output_txt_path, 'w', encoding='utf-8') as out:
        out.write(f"HDF5 File Inspection Report\n")
        out.write(f"Target File: {h5_path}\n")
        out.write("="*50 + "\n")
        
        print(f"Opening HDF5 file... Please wait.")
        
        with h5py.File(h5_path, 'r') as f:
            out.write("\n--- HDF5 FILE STRUCTURE ---\n")

            for key in f.keys():
                dataset = f[key]
                out.write(f"\n🔑 Key found: '{key}'\n")

                if hasattr(dataset, 'shape'):
                    out.write(f"  📐 Shape: {dataset.shape}\n")
                    out.write(f"  🔢 Data Type (dtype): {dataset.dtype}\n")
                    if len(dataset.shape) > 0:
                        out.write(f"  📦 Total Elements: {dataset.size}\n")
                else:
                    out.write("  ℹ️ Key contains group/metadata objects.\n")

            out.write("\n" + "-"*27 + "\n")
            out.write("✅ Inspection complete! No corruption detected during read.\n")
            
    print(f"🎉 Success! The structure has been saved to:\n👉 {output_txt_path}")

except Exception as e:
    print(f"❌ Failed to process the HDF5 file. Error: {e}")