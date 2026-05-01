import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ======================================
# FILE PATHS
# ======================================

DATA_FOLDER = "bot-iot"

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

print("CSV files found:", len(csv_files))

dfs = []

for file in csv_files:
    try:
        part = pd.read_csv(file, low_memory=False)
        print("Loaded:", file, part.shape)
        dfs.append(part)
    except Exception as e:
        print("Skipped:", file, e)

original_df = pd.concat(dfs, ignore_index=True, sort=False)

print("\nOriginal dataset shape:", original_df.shape)

# ======================================
# LOAD PROCESSED DATA
# ======================================

processed_file = "bot-iot_final.csv"
processed_df = pd.read_csv(processed_file)

print("Processed dataset shape:", processed_df.shape)

# ======================================
# AUTOMATIC LABEL DETECTION
# ======================================

# Hardcode original dataset label columns
orig_binary = 'attack'
orig_multi = 'category'

# Detect processed dataset label columns automatically
proc_binary = "binary_label"
proc_multi = "multi_label"

print("\nDetected Columns")
print("---------------------------")
print("Original Binary Label:", orig_binary)
print("Original Multiclass Label:", orig_multi)

print("Processed Binary Label:", proc_binary)
print("Processed Multiclass Label:", proc_multi)

# ======================================
# BINARY DISTRIBUTION
# ======================================

print("\n==============================")
print("BINARY CLASS DISTRIBUTION")
print("==============================")

orig_bin_counts = original_df[orig_binary].value_counts()
proc_bin_counts = processed_df[proc_binary].value_counts()

print("\nOriginal Binary Distribution")
print(orig_bin_counts)

print("\nProcessed Binary Distribution")
print(proc_bin_counts)

# ======================================
# MULTICLASS DISTRIBUTION
# ======================================

print("\n==============================")
print("MULTICLASS DISTRIBUTION")
print("==============================")

orig_multi_counts = original_df[orig_multi].value_counts()

print("\nOriginal Multiclass Distribution")
print(orig_multi_counts)

if proc_multi:
    proc_multi_counts = processed_df[proc_multi].value_counts()

    print("\nProcessed Multiclass Distribution")
    print(proc_multi_counts)

# ======================================
# PERCENTAGES
# ======================================

print("\n==============================")
print("PERCENTAGE DISTRIBUTION")
print("==============================")

print("\nOriginal Binary %")
print((orig_bin_counts / orig_bin_counts.sum()) * 100)

print("\nProcessed Binary %")
print((proc_bin_counts / proc_bin_counts.sum()) * 100)

print("\nOriginal Multiclass %")
print((orig_multi_counts / orig_multi_counts.sum()) * 100)

if proc_multi:
    print("\nProcessed Multiclass %")
    print((proc_multi_counts / proc_multi_counts.sum()) * 100)

# ======================================
# IMBALANCE RATIO
# ======================================

print("\n==============================")
print("IMBALANCE RATIO")
print("==============================")

def imbalance_ratio(counts):
    return counts.max() / counts.min()

print("Binary Original IR:", imbalance_ratio(orig_bin_counts))
print("Binary Processed IR:", imbalance_ratio(proc_bin_counts))

print("Multiclass Original IR:", imbalance_ratio(orig_multi_counts))

if proc_multi:
    print("Multiclass Processed IR:", imbalance_ratio(proc_multi_counts))

# ======================================
# VISUALIZATION
# ======================================


print("\nGenerating and saving plots...")
PLOT_DIR = "plots"

# Binary comparison
plt.figure(figsize=(6,5))
orig_bin_counts.sort_index().plot(kind="bar")
plt.title("Original Binary Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "original_binary_distribution.png"))
plt.close()

plt.figure(figsize=(6,5))
proc_bin_counts.sort_index().plot(kind="bar")
plt.title("Processed Binary Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "processed_binary_distribution.png"))
plt.close()

# Multiclass comparison
plt.figure(figsize=(8,5))
orig_multi_counts.plot(kind="bar")
plt.title("Original Multiclass Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "original_multiclass_distribution.png"))
plt.close()

if proc_multi:
    plt.figure(figsize=(8,5))
    proc_multi_counts.plot(kind="bar")
    plt.title("Processed Multiclass Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "processed_multiclass_distribution.png"))
    plt.close()

# ======================================
# SAVE COMBINED ORIGINAL DATASET
# ======================================

original_df.to_csv("combined_original_dataset.csv", index=False)

print("\nCombined original dataset saved.")