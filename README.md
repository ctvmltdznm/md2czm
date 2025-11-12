# MD-to-CZM: Extract Cohesive Zone Models from Molecular Dynamics

Automated pipeline to extract traction-separation laws from MD simulations of material interfaces for use in finite element cohesive zone modeling.

## Features

- **Automatic interface detection** in periodic MD trajectories
- **Trajectory unwrapping** for continuous separation tracking
- **Multiple CZM models**: Triangular, Exponential (Park-Paulino-Roesler), Bilinear Parabolic
- **Constrained fitting** ensures physically valid concave-down parabolas
- **Mode I (tensile) and Mode II/III (shear)** support
- **Auto-naming** outputs based on deformation mode

## Requirements

```bash
pip install numpy pandas scipy matplotlib ovito
```

## Workflow

```
LAMMPS trajectory → Track separation → Merge with stress → Extract CZM
  (.lammpstrj)         (Python/OVITO)    (Python)          (Python)
```

## Usage

### 1. Track Interface Separation

**Tensile mode:**
```bash
ovitos separation.py tensile_z.lammpstrj
# Output: tensile_z_separation_z.csv
```

**Shear mode:**
```bash
ovitos shear_separation.py shear_xz.lammpstrj z
# Output: shear_xz_shear_xz.csv
# (z = interface normal, tracks slip along x)
```

### 2. Merge with Stress Data

```bash
python merge_stress_separation.py tensile_z_separation_z.csv stress_interface_z.dat
# Output: stress_interface_z_with_separation.dat
# Might require trimming to include only first peak (if multiple)
```

### 3. Extract CZM Parameters

```bash
python extract_czm_parameters.py stress_interface_z_with_separation.dat --plot
# Default options for smoothing: --window 15 --poly 3 --plot
# Outputs:
#   - czm_parameters_z.dat (model parameters)
#   - czm_curve_z.png (comparison plot)
```

**Options:**
- `--units GPa` or `--units MPa` (default: GPa)
- `--window 15` smoothing window (default: 15)
- `--poly 3` polynomial order (default: 3)
- `--no-smooth` disable smoothing

## Output Files

### Separation CSV
```
frame,separation_z_A
0,18.234
1,18.456
...
```

### CZM Parameters (.dat)
```
# Common Parameters
K = 120339007.94  # MPa/mm - Initial stiffness
sigma_c = 176.520  # MPa - Peak strength
G_I = 9.762e-05  # N/mm - Fracture energy

# BILINEAR PARABOLIC MODEL (RECOMMENDED)
a1_load = -1.942e+15  # (< 0)
b1_load = -3.688e+13
a2_unload = -1.594e+13  # (< 0)
b2_unload = 3.688e+13
delta_peak = 2.254e-06  # mm
delta_max = 2.691e-06  # mm (element deletion)
```

## CZM Models

### 1. Triangular (Simple)
Linear loading → peak → linear unloading
- **Use for:** Quick estimates, simple FEM implementation
- **Limitation:** Poor fit for nonlinear behavior

### 2. Exponential (Park-Paulino-Roesler)
`T(δ) = (T_peak/e) * (δ/δ_n) * exp(1 - δ/δ_n)`
- **Use for:** Ductile interfaces, smooth softening
- **Requires:** Explicit cutoff δ_max in FEM

### 3. Bilinear Parabolic (Recommended)
Loading: `T = a₁(δ - δ_peak)² + b₁(δ - δ_peak) + T_peak`  
Unloading: `T = a₂(δ - δ_peak)² + b₂(δ - δ_peak) + T_peak`
- **Use for:** Best fit to MD data (typically R² > 0.9)
- **Features:** Different curvatures for loading/unloading
- **Constraint:** Both a₁ < 0 and a₂ < 0 enforced

## Key Algorithm Features

### Interface Detection
1. Identifies aragonite atoms (type ≤ 3)
2. Sorts along interface normal
3. Finds largest gap (accounts for periodic boundaries)
4. Selects bounding particles

### Trajectory Unwrapping
Frame-to-frame comparison:
```python
if displacement > L/2:  # Boundary crossing detected
    unwrapped_position -= L
```
Handles particles crossing periodic boundaries correctly.

### Constrained Fitting
Uses `scipy.optimize.least_squares` with bounds:
```python
bounds=([-inf, -inf], [-1e-12, inf])  # Force a < 0
```
Ensures parabolas are concave down (physical requirement).

## Example Output

![CZM Comparison](/example/czm_curve_xy.png)

- **Left:** Model comparison on MD data
- **Right:** Residual analysis (fit quality)

## File Naming Convention

Input file determines output names:
- `tensile_z.lammpstrj` → `czm_parameters_z.dat`, `czm_curve_z.png`
- `shear_xz.lammpstrj` → `czm_parameters_xz.dat`, `czm_curve_xz.png`

## Troubleshooting

**Problem:** Bilinear fit fails with "constraint failed: a ≥ 0"  
**Solution:** Data has long tail → increase smoothing (`--window 21`)

**Problem:** Negative separation values  
**Solution:** Atom crossed boundary → check OVITO visualization with reported IDs

**Problem:** Exponential model poor fit (R² < 0.5)  
**Solution:** Use bilinear parabolic model instead

**Problem:** "Lost track of atoms at frame X"  
**Solution:** LAMMPS reordered atoms → script finds by ID automatically

## Requirements

- Python 3.7+
- OVITO 3.x (for trajectory analysis)
- NumPy, Pandas, SciPy, Matplotlib
