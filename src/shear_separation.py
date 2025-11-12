#!/usr/bin/env python3
"""
Interface shear tracking script using OVITO API
Usage: ovitos track_shear_fixed.py shear_xz.lammpstrj z
"""
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
from ovito.io import import_file
import numpy as np
import pandas as pd
import sys
import re

# ---------- PARSE ARGUMENTS ----------
if len(sys.argv) < 3:
    print("Usage: ovitos track_shear_fixed.py shear_xz.lammpstrj z")
    print("  First arg: trajectory file")
    print("  Second arg: interface normal direction (x, y, or z)")
    sys.exit(1)

input_file = sys.argv[1]
interface_normal = sys.argv[2].lower()

if interface_normal not in ['x', 'y', 'z']:
    print(f"ERROR: Interface normal must be x, y, or z (got: {interface_normal})")
    sys.exit(1)

# Extract deformation mode from filename
match = re.search(r'([xyz]{2})', input_file, re.IGNORECASE)
deformation_mode = match.group(1).lower() if match else 'xy'

# Determine shear direction
mode_axes = set(deformation_mode)
normal_axis = {interface_normal}
shear_axes = mode_axes - normal_axis

if len(shear_axes) == 0:
    print(f"ERROR: Mode {deformation_mode} doesn't include shear perpendicular to normal {interface_normal}")
    sys.exit(1)

shear_direction = list(shear_axes)[0]

# Coordinate mapping
coord_map = {'x': 0, 'y': 1, 'z': 2}
normal_idx = coord_map[interface_normal]
shear_idx = coord_map[shear_direction]

print(f"Shear mode: {deformation_mode.upper()}")
print(f"Interface normal: {interface_normal.upper()}")
print(f"Tracking slip along: {shear_direction.upper()}")

# ---------- LOAD DATA ----------
pipe = import_file(input_file)
first = pipe.compute(0)

pos0 = np.array(first.particles['Position'])
types0 = np.array(first.particles['Particle Type'])
ids0 = np.array(first.particles['Particle Identifier'])
box0 = first.cell

L_normal = np.linalg.norm(box0[:, normal_idx])
L_shear = np.linalg.norm(box0[:, shear_idx])

print(f"Box: {interface_normal}={L_normal:.2f} A, {shear_direction}={L_shear:.2f} A")

# Find aragonite atoms
arag_mask = types0 <= 3
n_arag = np.sum(arag_mask)
print(f"Found {n_arag} aragonite atoms")

# Get positions along normal
coord_arag = pos0[arag_mask, normal_idx]
arag_indices = np.where(arag_mask)[0]
arag_ids = ids0[arag_mask]

# Sort and find largest gap
sorted_order = np.argsort(coord_arag)
coord_sorted = coord_arag[sorted_order]
idx_sorted = arag_indices[sorted_order]
ids_sorted = arag_ids[sorted_order]

diffs = np.diff(coord_sorted)
wrap_gap = (coord_sorted[0] + L_normal) - coord_sorted[-1]
all_gaps = np.hstack([diffs, wrap_gap])

imax = np.argmax(all_gaps)
gap_size = all_gaps[imax]
print(f"Interface gap: {gap_size:.2f} A")

# Select bounding particles
if imax < len(diffs):
    lower_idx = idx_sorted[imax]
    upper_idx = idx_sorted[imax + 1]
    lower_id = ids_sorted[imax]
    upper_id = ids_sorted[imax + 1]
else:
    lower_idx = idx_sorted[-1]
    upper_idx = idx_sorted[0]
    lower_id = ids_sorted[-1]
    upper_id = ids_sorted[0]

# Print initial positions
pos_low_0 = pos0[lower_idx]
pos_up_0 = pos0[upper_idx]

print(f"\nTracking particles:")
print(f"  Lower: ID={lower_id}, {interface_normal}={pos_low_0[normal_idx]:.2f}, {shear_direction}={pos_low_0[shear_idx]:.2f} A")
print(f"  Upper: ID={upper_id}, {interface_normal}={pos_up_0[normal_idx]:.2f}, {shear_direction}={pos_up_0[shear_idx]:.2f} A")

# Initial values (wrapped positions from frame 0)
n_low_0 = pos_low_0[normal_idx]
n_up_0 = pos_up_0[normal_idx]
s_low_0 = pos_low_0[shear_idx]
s_up_0 = pos_up_0[shear_idx]

# ---------- TRACK ALL FRAMES ----------
print(f"\nProcessing {pipe.source.num_frames} frames...")

frames = []
normal_seps = []
shear_slips = []

# Initialize unwrapped positions - start directly from frame 0 wrapped coordinates
n_low_unwrap = n_low_0
n_up_unwrap = n_up_0
s_low_unwrap = s_low_0
s_up_unwrap = s_up_0

# Initialize prev variables
n_low_prev = n_low_0
n_up_prev = n_up_0
s_low_prev = s_low_0
s_up_prev = s_up_0

for frame in range(pipe.source.num_frames):
    data = pipe.compute(frame)
    pos = np.array(data.particles['Position'])
    ids = np.array(data.particles['Particle Identifier'])
    box = data.cell
    
    L_n = np.linalg.norm(box[:, normal_idx])
    L_s = np.linalg.norm(box[:, shear_idx])
    
    # Find particles by ID
    lower_idx_frame = np.where(ids == lower_id)[0]
    upper_idx_frame = np.where(ids == upper_id)[0]
    
    if len(lower_idx_frame) == 0 or len(upper_idx_frame) == 0:
        print(f"WARNING: Lost track of atoms at frame {frame}!")
        normal_seps.append(np.nan)
        shear_slips.append(np.nan)
        frames.append(frame)
        continue
    
    idx_low = lower_idx_frame[0]
    idx_up = upper_idx_frame[0]
    
    # Current wrapped positions
    n_low_wrap = pos[idx_low, normal_idx]
    n_up_wrap = pos[idx_up, normal_idx]
    s_low_wrap = pos[idx_low, shear_idx]
    s_up_wrap = pos[idx_up, shear_idx]
    
    # Unwrap by detecting jumps > L/2 (boundary crossings)
    # CRITICAL: Always add displacement, then adjust for wrapping
    if frame > 0:
        # Normal direction - lower atom
        dn_low = n_low_wrap - n_low_prev
        n_low_unwrap += dn_low
        if dn_low > 0.5 * L_n:  # Wrapped backward
            n_low_unwrap -= L_n
        elif dn_low < -0.5 * L_n:  # Wrapped forward
            n_low_unwrap += L_n
        
        # Normal direction - upper atom
        dn_up = n_up_wrap - n_up_prev
        n_up_unwrap += dn_up
        if dn_up > 0.5 * L_n:
            n_up_unwrap -= L_n
        elif dn_up < -0.5 * L_n:
            n_up_unwrap += L_n
        
        # Shear direction - lower atom
        ds_low = s_low_wrap - s_low_prev
        s_low_unwrap += ds_low
        if ds_low > 0.5 * L_s:
            s_low_unwrap -= L_s
        elif ds_low < -0.5 * L_s:
            s_low_unwrap += L_s
        
        # Shear direction - upper atom
        ds_up = s_up_wrap - s_up_prev
        s_up_unwrap += ds_up
        if ds_up > 0.5 * L_s:
            s_up_unwrap -= L_s
        elif ds_up < -0.5 * L_s:
            s_up_unwrap += L_s
    
    # Store current wrapped positions for next iteration
    n_low_prev = n_low_wrap
    n_up_prev = n_up_wrap
    s_low_prev = s_low_wrap
    s_up_prev = s_up_wrap
    
    # Calculate separations from unwrapped coordinates
    normal_sep = n_up_unwrap - n_low_unwrap
    shear_slip = s_up_unwrap - s_low_unwrap
    
    frames.append(frame)
    normal_seps.append(normal_sep)
    shear_slips.append(shear_slip)

# ---------- NORMALIZE TO START FROM ZERO ----------
# Subtract initial offset from shear slip
initial_slip = shear_slips[0]
shear_slips_normalized = [s - initial_slip for s in shear_slips]

print(f"\nInitial normal separation: {normal_seps[0]:.2f} A")
print(f"Initial shear offset: {initial_slip:.2f} A (will be subtracted)")

# ---------- SAVE ----------
df = pd.DataFrame({
    'frame': frames,
    f'normal_sep_{interface_normal}_A': normal_seps,
    f'shear_slip_{shear_direction}_A': shear_slips_normalized
})

outname = input_file.replace('.lammpstrj', f'_separation.csv')
df.to_csv(outname, index=False)

print(f"\nSaved: {outname}")
print(f"  Normal separation: {normal_seps[0]:.2f} -> {normal_seps[-1]:.2f} A (Delta={normal_seps[-1]-normal_seps[0]:.2f})")
print(f"  Shear slip (normalized): {shear_slips_normalized[0]:.2f} -> {shear_slips_normalized[-1]:.2f} A")
print(f"  (Subtracted initial offset: {initial_slip:.2f} A)")
print(f"\nVerify in OVITO: IDs {lower_id} and {upper_id}")