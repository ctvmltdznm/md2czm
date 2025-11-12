# Interface separation tracking script using OVITO API
# Usage: ovitos track_separation.py tensile_y.lammpstrj
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
from ovito.io import import_file
from ovito.modifiers import *
import numpy as np
import pandas as pd
import sys
import re

# ---------- AUTO-DETECT INPUT AND DIRECTION ----------
input_file = sys.argv[1] if len(sys.argv) > 1 else 'tensile_z.lammpstrj'

# Determine deformation direction from filename
match = re.search(r'_(x|y|z|xy|yz|xz)\b', input_file)
if match:
    deformation_dir = match.group(1)
else:
    deformation_dir = 'z'  # default fallback

# Interface normal is the first letter (e.g., xy → x, z → z)
interface_axis = deformation_dir[0]
coord_map = {'x': 0, 'y': 1, 'z': 2}
coord_index = coord_map[interface_axis]
axis_label = ['x', 'y', 'z'][coord_index]

print(f"Detected deformation: {deformation_dir.upper()}")
print(f"Tracking interface normal: {interface_axis.upper()}-axis")

# ---------- IMPORT DATA ----------
pipe = import_file(input_file)

# Compute first frame
first = pipe.compute(0)
pos0 = np.array(first.particles['Position'])  # Already in real coordinates (Å)
types0 = np.array(first.particles['Particle Type'])
ids0 = np.array(first.particles['Particle Identifier'])  # LAMMPS atom IDs
box0 = first.cell

# Get box length along interface axis
L0 = np.linalg.norm(box0[:, coord_index])  # Å

print(f"Box length along {axis_label}-axis: {L0:.2f} Å")

# Select only aragonite atoms (type ≤ 3)
arag_mask = types0 <= 3
n_arag = np.sum(arag_mask)
print(f"Found {n_arag} aragonite atoms (types 1-3)")

# Get aragonite coordinates along interface axis
coord_arag = pos0[arag_mask, coord_index]
arag_array_indices = np.where(arag_mask)[0]  # Array indices where aragonite atoms are
arag_ids = ids0[arag_mask]  # Actual LAMMPS IDs

# Sort by position along interface axis
sorted_order = np.argsort(coord_arag)
coord_sorted = coord_arag[sorted_order]
idx_sorted_array = arag_array_indices[sorted_order]  # Array indices, sorted by position
ids_sorted = arag_ids[sorted_order]  # LAMMPS IDs, sorted by position

# Find largest gap (including wrap-around)
diffs = np.diff(coord_sorted)
wrap_gap = (coord_sorted[0] + L0) - coord_sorted[-1]  # Gap wrapping around periodic boundary
all_gaps = np.hstack([diffs, wrap_gap])

# Find the interface (largest gap)
imax = np.argmax(all_gaps)
gap_size = all_gaps[imax]

print(f"\nLargest gap detected: {gap_size:.2f} Å")

# Particles bounding the interface
if imax < len(diffs):
    # Gap is between two consecutive sorted particles
    lower_array_idx = idx_sorted_array[imax]
    upper_array_idx = idx_sorted_array[imax + 1]
    lower_lammps_id = ids_sorted[imax]
    upper_lammps_id = ids_sorted[imax + 1]
else:
    # Gap wraps around (between last and first)
    lower_array_idx = idx_sorted_array[-1]
    upper_array_idx = idx_sorted_array[0]
    lower_lammps_id = ids_sorted[-1]
    upper_lammps_id = ids_sorted[0]

# Sanity check: print their info at frame 0
t_low = int(types0[lower_array_idx])
t_up = int(types0[upper_array_idx])
c_low = pos0[lower_array_idx, coord_index]
c_up = pos0[upper_array_idx, coord_index]

print(f"\nSelected interface bounding particles:")
print(f"  Lower: LAMMPS_ID={lower_lammps_id}, type={t_low}, {axis_label}={c_low:.2f} Å")
print(f"  Upper: LAMMPS_ID={upper_lammps_id}, type={t_up}, {axis_label}={c_up:.2f} Å")
print(f"  (In OVITO, select these particles using their LAMMPS IDs: {lower_lammps_id} and {upper_lammps_id})")

if t_low > 3 or t_up > 3:
    print("WARNING: Selected particles are not aragonite (type>3). Check atom types!")

# Initial separation (accounting for PBC wrap)
raw_sep0 = c_up - c_low
if raw_sep0 < 0:  # Wrapped across boundary
    initial_sep = raw_sep0 + L0
else:
    initial_sep = raw_sep0
    
print(f"Initial separation: {initial_sep:.2f} Å")

# ---------- TRACK SEPARATION OVER ALL FRAMES ----------
print(f"\nTracking {pipe.source.num_frames} frames...")

frames = []
separations = []

for frame in range(pipe.source.num_frames):
    data = pipe.compute(frame)
    pos = np.array(data.particles['Position'])  # Already in Å
    ids = np.array(data.particles['Particle Identifier'])  # LAMMPS IDs this frame
    box = data.cell
    
    # Box length along interface axis
    L = np.linalg.norm(box[:, coord_index])
    
    # CRITICAL: Find the array indices of our tracked atoms by their LAMMPS ID
    # (atoms can be reordered between frames in the dump file!)
    lower_idx_this_frame = np.where(ids == lower_lammps_id)[0]
    upper_idx_this_frame = np.where(ids == upper_lammps_id)[0]
    
    # Verify we found them
    if len(lower_idx_this_frame) == 0 or len(upper_idx_this_frame) == 0:
        print(f"WARNING: Lost track of atoms at frame {frame}!")
        separations.append(np.nan)
        frames.append(frame)
        continue
    
    lower_idx = lower_idx_this_frame[0]
    upper_idx = upper_idx_this_frame[0]
    
    # Get positions of the two tracked particles
    z_low = pos[lower_idx, coord_index]
    z_up = pos[upper_idx, coord_index]
    
    # Calculate separation (raw)
    dz = z_up - z_low
    
    # Unwrap if needed (they should stay on opposite sides of interface)
    # If separation is negative, interface wraps around periodic boundary
    if dz < 0:
        dz += L
    # If separation is unreasonably large, might have wrapped the other way
    elif dz > 0.75 * L:  # Threshold: if more than 75% of box, likely wrapped wrong
        dz -= L
    
    frames.append(frame)
    separations.append(dz)

# ---------- SAVE RESULTS ----------
df = pd.DataFrame({
    'frame': frames,
    f'separation_{axis_label}_A': separations
})

outname = input_file.replace('.lammpstrj', f'_separation.csv')
df.to_csv(outname, index=False)

print(f"\n✓ Saved: {outname}")
print(f"  Initial separation: {separations[0]:.2f} Å")
print(f"  Final separation: {separations[-1]:.2f} Å")
print(f"  Change: {separations[-1] - separations[0]:.2f} Å")
print(f"\nTo verify in OVITO: Select particles with IDs {lower_lammps_id} and {upper_lammps_id}")