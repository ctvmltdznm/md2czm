#!/usr/bin/env python3
"""
Merge separation/shear data with stress-strain data
Usage:
  python merge_stress_separation.py separation_z.csv stress_strain.dat
  python merge_stress_separation.py shear_xz.csv stress_strain.dat
"""

import pandas as pd
import sys

if len(sys.argv) < 3:
    print("Usage: python merge_stress_separation.py <separation_csv> <stress_strain_dat>")
    print("Examples:")
    print("  python merge_stress_separation.py tensile_z_separation_z.csv stress_strain_z.dat")
    print("  python merge_stress_separation.py shear_xz_shear_xz.csv stress_strain_xz.dat")
    sys.exit(1)

separation_file = sys.argv[1]
stress_file = sys.argv[2]

print(f"Reading separation data: {separation_file}")
print(f"Reading stress-strain data: {stress_file}")

# Read separation/shear data (CSV)
sep_data = pd.read_csv(separation_file)

# Check if it's tensile or shear based on columns
columns = sep_data.columns.tolist()
is_shear = any('shear_slip' in col for col in columns)

if is_shear:
    print("Detected: SHEAR mode")
    # Find the shear slip column
    shear_col = [col for col in columns if 'shear_slip' in col][0]
    
    # Normalize shear slip to start from zero
    initial_slip = sep_data[shear_col].iloc[0]
    sep_data[shear_col] = sep_data[shear_col] - initial_slip
    print(f"  Normalized {shear_col} (subtracted initial value: {initial_slip:.2f})")
else:
    print("Detected: TENSILE mode")

# Read stress-strain data (space-separated, skip first line which is header)
stress_data = pd.read_csv(stress_file, sep=r'\s+', skiprows=1, header=None,
                          names=['Eng_strain', 'True_strain', 'S11', 'S22', 'S33', 'S12', 'S13', 'S23'])

print(f"\nData shapes:")
print(f"  Separation data: {sep_data.shape[0]} frames")
print(f"  Stress-strain data: {stress_data.shape[0]} frames")

# Check if lengths match
if sep_data.shape[0] != stress_data.shape[0]:
    print("\nWARNING: Number of frames don't match!")
    print("  Using minimum length for merging")
    min_len = min(sep_data.shape[0], stress_data.shape[0])
    sep_data = sep_data.iloc[:min_len]
    stress_data = stress_data.iloc[:min_len]

# Merge data
# Drop 'frame' column from separation data as we're merging by index
sep_data_no_frame = sep_data.drop('frame', axis=1)

# Combine
merged = pd.concat([stress_data, sep_data_no_frame], axis=1)

# Generate output filename
if is_shear:
    outname = stress_file.replace('.dat', '_with_separation.dat')
else:
    outname = stress_file.replace('.dat', '_with_separation.dat')

# Save as space-separated
merged.to_csv(outname, sep=' ', index=False, float_format='%.6f')

print(f"\n✓ Saved merged data: {outname}")
print(f"  Columns: {list(merged.columns)}")
print(f"  Total rows: {merged.shape[0]}")

# Show first few rows
print("\nFirst 3 rows:")
print(merged.head(3).to_string(index=False))