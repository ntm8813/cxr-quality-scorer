import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_distributions():
    csv_path = "data/processed/degradation_manifest.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: Cannot find {csv_path}. Run the orchestrator first.")
        return
        
    df = pd.read_csv(csv_path)
    
    print("\n=============================================")
    print("   LABELED DATASET DISTRIBUTION SUMMARY   ")
    print("=============================================")
    print(f"Total Array Rows Generated: {len(df)}")
    print("\n🔹 Breakdown per Degradation Axis:")
    print(df['axis'].value_counts())
    print("\n🔹 Breakdown per Severity Category:")
    print(df['severity'].value_counts())
    print("=============================================\n")
    
    # Generate distribution bar graph
    plt.figure(figsize=(8, 5))
    df['severity'].value_counts().sort_index().plot(kind='bar', color=['#4CAF50', '#FF9800', '#F44336'])
    plt.title("CXR Labeled Dataset Balance Across Severity Levels")
    plt.xlabel("Severity (0=Clean, 1=Borderline, 2=Repeat)")
    plt.ylabel("Image Count")
    plt.xticks(rotation=0)
    
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/severity_distribution.png")
    print("✅ Balance distribution plot successfully exported to:")
    print("   reports/figures/severity_distribution.png")

if __name__ == "__main__":
    plot_distributions()