#!/usr/bin/env python3
"""
Extract cohesive zone model (CZM) parameters from traction-separation data
Handles both Mode I (tensile) and Mode II/III (shear)

Usage:
  Auto-detect: python extract_czm_clean.py merged_data.txt --plot
  Explicit:    python extract_czm_clean.py merged_data.txt --mode shear --stress S23 --separation shear_slip_y_A
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, least_squares
import argparse
import sys
import os
import re

def detect_mode_and_columns(df):
    """Auto-detect test mode and relevant columns"""
    cols = df.columns.tolist()
    
    # Check for separation columns
    sep_cols = [c for c in cols if 'sep' in c.lower() or 'slip' in c.lower()]
    
    if any('shear_slip' in c for c in sep_cols):
        mode = 'shear'
        sep_col = [c for c in cols if 'shear_slip' in c][0]
        
        # Determine stress column from shear slip direction
        shear_match = re.search(r'shear_slip_([xyz])', sep_col, re.IGNORECASE)
        normal_match = re.search(r'normal_sep_([xyz])', ' '.join(cols), re.IGNORECASE)
        
        if shear_match:
            shear_dir = shear_match.group(1).lower()
            normal_dir = normal_match.group(1).lower() if normal_match else None
            
            # Map to stress component: S_ij where i=normal, j=shear
            stress_map = {
                ('x', 'y'): 'S12', ('y', 'x'): 'S12',
                ('x', 'z'): 'S13', ('z', 'x'): 'S13',
                ('y', 'z'): 'S23', ('z', 'y'): 'S23',
            }
            
            if normal_dir:
                stress_col = stress_map.get((normal_dir, shear_dir)) or stress_map.get((shear_dir, normal_dir))
            else:
                # Fallback
                if shear_dir == 'x':
                    stress_col = 'S13' if 'S13' in cols else 'S12'
                elif shear_dir == 'y':
                    stress_col = 'S23' if 'S23' in cols else 'S12'
                elif shear_dir == 'z':
                    stress_col = 'S23' if 'S23' in cols else 'S13'
                else:
                    stress_col = None
            
            if stress_col not in cols:
                for sc in ['S23', 'S13', 'S12']:
                    if sc in cols:
                        stress_col = sc
                        break
        else:
            stress_col = None
            for sc in ['S23', 'S13', 'S12']:
                if sc in cols:
                    stress_col = sc
                    break
    else:
        mode = 'tensile'
        sep_col = [c for c in cols if 'normal_sep' in c or 'separation' in c][0]
        
        if 'x' in sep_col.lower():
            stress_col = 'S11'
        elif 'y' in sep_col.lower():
            stress_col = 'S22'
        elif 'z' in sep_col.lower():
            stress_col = 'S33'
        else:
            stress_col = 'S33'
    
    return mode, stress_col, sep_col

def smooth_data(data, window=15, polyorder=3):
    """Apply Savitzky-Golay filter"""
    if len(data) < window:
        window = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window < 3:
            return data
    return savgol_filter(data, window, polyorder)

def calculate_initial_stiffness(delta, T, fraction=0.05, min_points=3):
    """Calculate initial stiffness from linear region"""
    N0 = max(min_points, int(len(delta) * fraction))
    N0 = min(N0, len(delta))
    K = np.polyfit(delta[:N0], T[:N0], 1)[0]
    return K

def fit_exponential_model(delta, T, T_peak):
    """Fit exponential (Park-Paulino-Roesler) model"""
    def exp_model(delta, delta_n):
        return (T_peak / np.e) * (delta / delta_n) * np.exp(1 - delta / delta_n)
    
    idx_peak = np.argmax(T)
    delta_peak = delta[idx_peak]
    delta_n_guess = delta_peak
    
    try:
        popt, _ = curve_fit(exp_model, delta, T, p0=[delta_n_guess], 
                           bounds=([delta[1]], [delta[-1]*2]), maxfev=10000)
        delta_n = popt[0]
        fitted_T = exp_model(delta, delta_n)
        G = T_peak * delta_n
        rmse = np.sqrt(np.mean((T - fitted_T)**2))
        ss_res = np.sum((T - fitted_T)**2)
        ss_tot = np.sum((T - np.mean(T))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'delta_n': delta_n,
            'G': G,
            'fitted_T': fitted_T,
            'rmse': rmse,
            'r_squared': r_squared,
            'success': True
        }
    except:
        return {'success': False}

def fit_bilinear_polynomial_model(delta, T, T_peak):
    """
    Fit bilinear parabolic model with STRICT constraint: a1 < 0 AND a2 < 0
    Both phases: T = a*(delta - delta_peak)^2 + b*(delta - delta_peak) + T_peak
    """
    idx_peak = np.argmax(T)
    delta_peak = delta[idx_peak]
    
    loading_mask = delta <= delta_peak
    unloading_mask = delta > delta_peak
    
    delta_load = delta[loading_mask]
    T_load = T[loading_mask]
    delta_unload = delta[unloading_mask]
    T_unload = T[unloading_mask]
    
    if len(delta_load) < 3:
        return {'success': False, 'error': f'Insufficient loading data ({len(delta_load)} points)'}
    if len(delta_unload) < 3:
        return {'success': False, 'error': f'Insufficient unloading data ({len(delta_unload)} points)'}
    
    try:
        delta_load_shifted = delta_load - delta_peak
        delta_unload_shifted = delta_unload - delta_peak
        T_load_shifted = T_load - T_peak
        T_unload_shifted = T_unload - T_peak
        
        # Loading fit with constraint a1 < 0
        def residuals_load(params):
            a1, b1 = params
            return T_load_shifted - (a1 * delta_load_shifted**2 + b1 * delta_load_shifted)
        
        # Initial guess - force negative
        coeffs_init = np.polyfit(delta_load_shifted, T_load_shifted, 2)
        a1_init = min(coeffs_init[0], -1e-10)  # Ensure negative
        b1_init = coeffs_init[1]
        
        # Constrain: -inf < a1 <= -1e-12, -inf < b1 < inf
        result_load = least_squares(
            residuals_load, 
            [a1_init, b1_init],
            bounds=([-np.inf, -np.inf], [-1e-12, np.inf]),  # a1 MUST be < -1e-12
            method='trf'
        )
        a1, b1 = result_load.x
        
        # Verify
        if a1 >= -1e-12:
            return {'success': False, 'error': f'Loading constraint failed: a1={a1:.2e} >= 0'}
        
        # Unloading fit with constraint a2 < 0
        def residuals_unload(params):
            a2, b2 = params
            return T_unload_shifted - (a2 * delta_unload_shifted**2 + b2 * delta_unload_shifted)
        
        coeffs_init_unload = np.polyfit(delta_unload_shifted, T_unload_shifted, 2)
        a2_init = min(coeffs_init_unload[0], -1e-20)
        b2_init = coeffs_init_unload[1]
        
        result_unload = least_squares(
            residuals_unload,
            [a2_init, b2_init],
            bounds=([-np.inf, -np.inf], [-1e-22, np.inf]),
            method='trf'
        )
        a2, b2 = result_unload.x
        
        # Verify
        if a2 >= -1e-22:
            return {'success': False, 'error': f'Unloading constraint failed: a2={a2:.2e} >= 0'}
        
        # Generate fitted values
        fitted_T = np.zeros_like(delta)
        fitted_T[loading_mask] = a1 * delta_load_shifted**2 + b1 * delta_load_shifted + T_peak
        fitted_T[unloading_mask] = a2 * delta_unload_shifted**2 + b2 * delta_unload_shifted + T_peak
        
        G = np.trapz(np.abs(fitted_T), delta)
        rmse = np.sqrt(np.mean((T - fitted_T)**2))
        ss_res = np.sum((T - fitted_T)**2)
        ss_tot = np.sum((T - np.mean(T))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate delta_max
        if abs(a2) < 1e-9:
            # Nearly linear case: T = b2*x + T_peak
            # Solve: b2*x + T_peak = 0  =>  x = -T_peak/b2
            if abs(b2) > 1e-12:
                x_max = -T_peak / b2
                if x_max > 0:
                    delta_f_calc = delta_peak + x_max
                else:
                    delta_f_calc = np.nan
            else:
                # Degenerate case: nearly constant
                delta_f_calc = np.nan
        else:
            # Quadratic case
            discriminant = b2**2 - 4*a2*T_peak
            
            if discriminant >= 0 and a2 < 0:
                x1 = (-b2 + np.sqrt(discriminant)) / (2*a2)
                x2 = (-b2 - np.sqrt(discriminant)) / (2*a2)
                # Take only POSITIVE root (unloading side: delta > delta_peak)
                positive_roots = [x for x in [x1, x2] if x > 0]
                x_max = max(positive_roots) if positive_roots else np.nan
                delta_f_calc = delta_peak + x_max if not np.isnan(x_max) else np.nan
            else:
                delta_f_calc = np.nan

#        discriminant = b2**2 - 4*a2*T_peak
#        if discriminant >= 0 and a2 < 0:
#            x1 = (-b2 + np.sqrt(discriminant)) / (2*a2)
#            x2 = (-b2 - np.sqrt(discriminant)) / (2*a2)
#            print(x1,x2)
#            # Take only the POSITIVE root (delta > delta_peak)
#            positive_roots = [x for x in [x1, x2] if x > 0]
#            x_max = max(positive_roots) if positive_roots else np.nan
#            delta_f_calc = delta_peak + x_max
#        else:
#            delta_f_calc = np.nan
        
        return {
            'a1': a1,
            'b1': b1,
            'a2': a2,
            'b2': b2,
            'delta_peak': delta_peak,
            'T_peak': T_peak,
            'delta_f': delta_f_calc,
            'G': G,
            'fitted_T': fitted_T,
            'rmse': rmse,
            'r_squared': r_squared,
            'n_load': len(delta_load),
            'n_unload': len(delta_unload),
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': f'Fitting error: {str(e)[:100]}'}

def calculate_czm_parameters(delta_A, T_raw, mode='tensile', smooth=True, 
                            smooth_window=15, smooth_poly=3, stiffness_fraction=0.05):
    """Calculate CZM parameters"""
    delta_mm = delta_A * 1e-7
    T_MPa = T_raw.copy()
    
    if smooth:
        T_smoothed = smooth_data(T_MPa, window=smooth_window, polyorder=smooth_poly)
    else:
        T_smoothed = T_MPa
    
    order = np.argsort(delta_mm)
    delta = delta_mm[order]
    T = T_smoothed[order]
    T_raw_sorted = T_MPa[order]
    
    results = {}
    
    # 1. Initial stiffness
    K = calculate_initial_stiffness(delta, T, fraction=stiffness_fraction)
    results['K'] = K
    results['K_units'] = 'MPa/mm'
    
    # 2. Peak strength
    T_peak = np.max(np.abs(T))
    if mode == 'tensile':
        results['sigma_c'] = T_peak
        results['strength_label'] = 'sigma_c'
    else:
        results['tau_c'] = T_peak
        results['strength_label'] = 'tau_c'
    results['strength_value'] = T_peak
    results['strength_units'] = 'MPa'
    
    # 3. Fracture energy
    G = np.trapz(np.abs(T), delta)
    if mode == 'tensile':
        results['G_I'] = G
        results['energy_label'] = 'G_I'
    else:
        results['G_II'] = G
        results['energy_label'] = 'G_II'
    results['energy_value'] = G
    results['energy_units'] = 'N/mm'
    
    # 4. Triangular model (starts from first data point)
    delta_min = delta[0]
    if T_peak > 0:
        delta_c = delta_min + 2.0 * G / T_peak
    else:
        delta_c = np.nan
    results['delta_c'] = delta_c
    results['delta_c_units'] = 'mm'
    results['delta_min'] = delta_min
    
    # 5. Max separation in data
    results['delta_max'] = delta[-1]
    results['delta_max_units'] = 'mm'
    
    # 6. Exponential model
    exp_fit = fit_exponential_model(delta, T, T_peak)
    results['exponential'] = exp_fit
    
    # 7. Bilinear parabolic model
    bilin_fit = fit_bilinear_polynomial_model(delta, T, T_peak)
    results['bilinear'] = bilin_fit
    
    # Store data
    results['delta_processed'] = delta
    results['T_processed'] = T
    results['T_raw_processed'] = T_raw_sorted
    results['smoothed'] = smooth
    results['mode'] = mode
    
    return results

def print_results(results):
    """Print CZM parameters"""
    mode = results['mode']
    
    print("\n" + "="*70)
    print(f"COHESIVE ZONE MODEL PARAMETERS - {mode.upper()} MODE")
    print("="*70)
    
    print(f"\n1. Initial Stiffness:")
    print(f"   K = {results['K']:.2f} {results['K_units']}")
    
    print(f"\n2. Peak Strength:")
    label = results['strength_label']
    print(f"   {label} = {results['strength_value']:.3f} {results['strength_units']}")
    
    print(f"\n3. Fracture Energy (from MD data):")
    label = results['energy_label']
    print(f"   {label} = {results['energy_value']:.3e} {results['energy_units']}")
    
    print(f"\n4. Maximum Separation in Data:")
    print(f"   delta_max = {results['delta_max']:.3e} {results['delta_max_units']}")
    
    final_traction = results['T_processed'][-1]
    complete_failure = abs(final_traction) < 0.1 * results['strength_value']
    
    if not complete_failure:
        print("\n   ! WARNING: Final traction still significant.")
        print("     Fracture energy may be underestimated.")
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Triangular
    print(f"\n+-- TRIANGULAR MODEL")
    print(f"|   Starts from delta_min = {results.get('delta_min', 0):.3e} mm")
    print(f"|   delta_c = {results['delta_c']:.3e} mm")
    print(f"|   G = {results['energy_value']:.3e} N/mm")
    print(f"+-- Simple linear approximation")
    
    # Exponential
    exp_fit = results['exponential']
    if exp_fit['success']:
        print(f"\n+-- EXPONENTIAL MODEL (Park-Paulino-Roesler)")
        print(f"|   delta_n = {exp_fit['delta_n']:.3e} mm")
        print(f"|   G = {exp_fit['G']:.3e} N/mm")
        print(f"|   R^2 = {exp_fit['r_squared']:.4f}")
        print(f"|   RMSE = {exp_fit['rmse']:.3f} MPa")
        print(f"+-- T = (T_peak/e) * (delta/delta_n) * exp(1 - delta/delta_n)")
    else:
        print(f"\n+-- EXPONENTIAL MODEL")
        print(f"+-- x Fitting failed")
    
    # Bilinear
    bilin_fit = results['bilinear']
    if bilin_fit['success']:
        a1, b1 = bilin_fit['a1'], bilin_fit['b1']
        a2, b2 = bilin_fit['a2'], bilin_fit['b2']
        T_pk = bilin_fit['T_peak']
        delta_f = bilin_fit['delta_f']
        print(f"\n+-- BILINEAR PARABOLIC MODEL (Both parabolas concave down)")
        print(f"|   Both phases: T = a*(delta - delta_p)^2 + b*(delta - delta_p) + T_p")
        print(f"|   delta_peak = {bilin_fit['delta_peak']:.3e} mm, T_peak = {T_pk:.2f} MPa")
        print(f"|   Loading: a1 = {a1:.3e} (< 0), b1 = {b1:.3e}")
        print(f"|   Unloading: a2 = {a2:.3e} (< 0), b2 = {b2:.3e}")
#        if not np.isnan(delta_f):
#            print(f"|   delta_max (calculated, T=0): {delta_f:.3e} mm")
        print(f"|   delta_max (calculated, T=0): {delta_f:.3e} mm")
        print(f"|   G = {bilin_fit['G']:.3e} N/mm")
        print(f"|   R^2 = {bilin_fit['r_squared']:.4f}")
        print(f"|   RMSE = {bilin_fit['rmse']:.3f} MPa")
        print(f"|   Points: {bilin_fit['n_load']} loading, {bilin_fit['n_unload']} unloading")
        print(f"+-- Both concave down (vertex at peak)")
    else:
        print(f"\n+-- BILINEAR PARABOLIC MODEL")
        error_msg = bilin_fit.get('error', 'Unknown error')
        print(f"+-- x Fitting failed: {error_msg}")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    models = []
    if exp_fit['success']:
        models.append(('Exponential', exp_fit['r_squared'], exp_fit))
    if bilin_fit['success']:
        models.append(('Bilinear', bilin_fit['r_squared'], bilin_fit))
    
    if models:
        best_model = max(models, key=lambda x: x[1])
        model_name, r2, model_data = best_model
        
        print(f"\n+ {model_name.upper()} model recommended (best fit)")
        print(f"  - Highest R^2 = {r2:.4f}")
        
        if model_name == 'Exponential':
            print(f"  - Physically motivated for ductile interfaces")
            print(f"  - Single parameter (delta_n) easy to calibrate")
        else:
            print(f"  - Parabolic hardening AND softening")
            print(f"  - Different curvatures for loading vs unloading")
            print(f"  - Represents elastic energy storage and release")
    else:
        print("\n! Use TRIANGULAR model (simple approximation)")
    
    print("="*70 + "\n")

def save_parameters(results, filename):
    """Save parameters to file"""
    input_basename = filename.split('_')[-1].split('.')[0] if '_' in filename else 'data'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# CZM Parameters - {results['mode'].upper()} Mode - {input_basename.upper()}\n")
        f.write(f"# Generated from traction-separation data\n")
        f.write("#" + "="*68 + "\n\n")
        
        f.write("# Common Parameters\n")
        f.write(f"K = {results['K']:.6f}  # {results['K_units']} - Initial stiffness\n")
        f.write(f"{results['strength_label']} = {results['strength_value']:.6f}  # {results['strength_units']}\n")
        f.write(f"{results['energy_label']}_MD = {results['energy_value']:.3e}  # {results['energy_units']}\n")
        f.write(f"delta_max = {results['delta_max']:.3e}  # {results['delta_max_units']}\n\n")
        
        f.write("#" + "="*68 + "\n")
        f.write("# TRIANGULAR MODEL\n")
        f.write("#" + "="*68 + "\n")
        f.write(f"delta_min_tri = {results.get('delta_min', 0):.3e}  # mm\n")
        f.write(f"delta_c_tri = {results['delta_c']:.3e}  # mm\n")
        f.write(f"G_tri = {results['energy_value']:.3e}  # N/mm\n\n")
        
        exp_fit = results['exponential']
        if exp_fit['success']:
            f.write("#" + "="*68 + "\n")
            f.write("# EXPONENTIAL MODEL\n")
            f.write("#" + "="*68 + "\n")
            f.write(f"delta_n_exp = {exp_fit['delta_n']:.3e}  # mm\n")
            f.write(f"G_exp = {exp_fit['G']:.3e}  # N/mm\n")
            f.write(f"R_squared_exp = {exp_fit['r_squared']:.6f}\n")
            f.write(f"RMSE_exp = {exp_fit['rmse']:.6f}  # MPa\n\n")
        
        bilin_fit = results['bilinear']
        if bilin_fit['success']:
            a1, b1 = bilin_fit['a1'], bilin_fit['b1']
            a2, b2 = bilin_fit['a2'], bilin_fit['b2']
            T_pk = bilin_fit['T_peak']
            delta_f = bilin_fit['delta_f']
            f.write("#" + "="*68 + "\n")
            f.write("# BILINEAR PARABOLIC MODEL\n")
            f.write("#" + "="*68 + "\n")
            f.write(f"# T = a*(delta - delta_peak)^2 + b*(delta - delta_peak) + T_peak\n")
            f.write(f"a1_load = {a1:.6e}  # (< 0)\n")
            f.write(f"b1_load = {b1:.6e}\n")
            f.write(f"a2_unload = {a2:.6e}  # (< 0)\n")
            f.write(f"b2_unload = {b2:.6e}\n")
            f.write(f"delta_peak_bilin = {bilin_fit['delta_peak']:.3e}  # mm\n")
            f.write(f"T_peak_bilin = {T_pk:.6f}  # MPa\n")
#            if not np.isnan(delta_f):
#                f.write(f"delta_max_bilin = {delta_f:.3e}  # mm (T=0)\n")
            f.write(f"delta_max_bilin = {delta_f:.3e}  # mm (T=0)\n")
            f.write(f"G_bilin = {bilin_fit['G']:.3e}  # N/mm\n")
            f.write(f"R_squared_bilin = {bilin_fit['r_squared']:.6f}\n")
            f.write(f"RMSE_bilin = {bilin_fit['rmse']:.6f}  # MPa\n\n")
        
        f.write("#" + "="*68 + "\n")
        f.write("# For FEM Implementation:\n")
        f.write("#" + "="*68 + "\n")
        
        best_r2 = 0
        best_model = None
        if exp_fit['success'] and exp_fit['r_squared'] > best_r2:
            best_model = 'exponential'
            best_r2 = exp_fit['r_squared']
        if bilin_fit['success'] and bilin_fit['r_squared'] > best_r2:
            best_model = 'bilinear'
            best_r2 = bilin_fit['r_squared']
        
        if best_model == 'exponential':
            f.write("# RECOMMENDED: Use Exponential model\n")
            f.write(f"# T = {results['strength_value']:.3f}/e * (delta/{exp_fit['delta_n']:.3e}) * exp(1 - delta/{exp_fit['delta_n']:.3e})\n")
            f.write(f"# Use cutoff: delta_max = {results['delta_max']:.3e} mm\n")
        elif best_model == 'bilinear':
            a1, b1 = bilin_fit['a1'], bilin_fit['b1']
            a2, b2 = bilin_fit['a2'], bilin_fit['b2']
            dp = bilin_fit['delta_peak']
            Tp = bilin_fit['T_peak']
            df = bilin_fit['delta_f']
            f.write("# RECOMMENDED: Use Bilinear Parabolic model\n")
            f.write(f"# Loading (delta <= {dp:.3e}):\n")
            f.write(f"#   T = {a1:.6e}*(delta - {dp:.3e})^2 + {b1:.6e}*(delta - {dp:.3e}) + {Tp:.3f}\n")
            f.write(f"# Unloading (delta > {dp:.3e}):\n")
            f.write(f"#   T = {a2:.6e}*(delta - {dp:.3e})^2 + {b2:.6e}*(delta - {dp:.3e}) + {Tp:.3f}\n")
            if not np.isnan(df):
                f.write(f"# Element deletion: delta >= {df:.3e} mm\n")
        else:
            f.write("# FALLBACK: Use Triangular model\n")
    
    print(f"+ Saved: {filename}")

def plot_traction_separation(results, save_plot=None):
    """Plot traction-separation curve"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available - skipping plot")
        return
    
    delta = results['delta_processed']
    T_smooth = results['T_processed']
    T_raw = results['T_raw_processed']
    mode = results['mode']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Models
    ax1.plot(delta, T_raw, 'o', color='lightblue', markersize=3, alpha=0.4, 
            label='Raw MD data', zorder=1)
    ax1.plot(delta, T_smooth, 'b-', linewidth=3, label='Smoothed MD data', zorder=2)
    
    # Triangular
    delta_c = results['delta_c']
    delta_min = results.get('delta_min', 0)
    T_peak = results['strength_value']
    delta_tri = np.array([delta_min, (delta_min + delta_c)/2, delta_c])
    T_tri = np.array([0, T_peak, 0])
    ax1.plot(delta_tri, T_tri, '--', color='red', linewidth=2, 
            label='Triangular', alpha=0.8, zorder=3)
    
    # Exponential
    exp_fit = results['exponential']
    if exp_fit['success']:
        ax1.plot(delta, exp_fit['fitted_T'], '-.', color='green', linewidth=2,
                label=f'Exponential (R^2={exp_fit["r_squared"]:.3f})', zorder=3)
    
    # Bilinear
    bilin_fit = results['bilinear']
    if bilin_fit['success']:
        ax1.plot(delta, bilin_fit['fitted_T'], ':', color='purple', linewidth=2.5,
                label=f'Bilinear Parabolic (R^2={bilin_fit["r_squared"]:.3f})', zorder=3)
    
    # Peak
    idx_peak = np.argmax(np.abs(T_smooth))
    ax1.plot(delta[idx_peak], T_smooth[idx_peak], 'ko', markersize=10, 
            markerfacecolor='red', markeredgewidth=2, label=f'Peak: {T_peak:.2f} MPa', zorder=5)
    
    # Labels
    if mode == 'tensile':
        ax1.set_ylabel('Normal Traction (MPa)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Normal Separation (mm)', fontsize=13, fontweight='bold')
        title1 = 'Mode I Traction-Separation: Model Comparison'
    else:
        ax1.set_ylabel('Shear Traction (MPa)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Tangential Slip (mm)', fontsize=13, fontweight='bold')
        title1 = 'Mode II/III Traction-Separation: Model Comparison'
    
    ax1.set_title(title1, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, loc='best', framealpha=0.95)
#    ax1.set_xlim(left=-0.02*delta[-1], right=delta[-1]*1.05)
    ax1.set_xlim(left=delta[0]*0.95, right=delta[-1]*1.05)  

    # Right plot: Residuals
    residuals_tri = T_smooth - np.interp(delta, delta_tri, T_tri, left=0, right=0)
    ax2.plot(delta, residuals_tri, '-', color='red', linewidth=1.5, 
            label='Triangular', alpha=0.7)
    
    if exp_fit['success']:
        residuals_exp = T_smooth - exp_fit['fitted_T']
        ax2.plot(delta, residuals_exp, '-.', color='green', linewidth=1.5,
                label=f'Exponential (RMSE={exp_fit["rmse"]:.2f})')
    
    if bilin_fit['success']:
        residuals_bilin = T_smooth - bilin_fit['fitted_T']
        ax2.plot(delta, residuals_bilin, ':', color='purple', linewidth=2,
                label=f'Bilinear Parabolic (RMSE={bilin_fit["rmse"]:.2f})')
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Separation (mm)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Residual (MPa)', fontsize=13, fontweight='bold')
    ax2.set_title('Model Residuals (MD data - Model)', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='best', framealpha=0.95)
    ax2.set_xlim(left=delta[0]*0.95, right=delta[-1]*1.05)
    
    # Text box
    textstr = f"Initial Stiffness:\n  K = {results['K']:.1f} MPa/mm\n\n"
    textstr += f"Peak Strength:\n  {results['strength_label']} = {results['strength_value']:.2f} MPa\n\n"
    textstr += f"Fracture Energy (MD):\n  {results['energy_label']} = {results['energy_value']:.3e} N/mm\n\n"
    
    if exp_fit['success']:
        textstr += f"Exponential:\n  delta_n = {exp_fit['delta_n']:.3e} mm\n"
        textstr += f"  G = {exp_fit['G']:.3e} N/mm\n\n"
    
    if bilin_fit['success']:
        textstr += f"Bilinear Parabolic:\n  delta_peak = {bilin_fit['delta_peak']:.3e} mm\n"
        textstr += f"  a1={bilin_fit['a1']:.2e}, a2={bilin_fit['a2']:.2e}\n"
        textstr += f"  G = {bilin_fit['G']:.3e} N/mm"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"+ Plot saved to: {save_plot}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extract CZM parameters')
    parser.add_argument('input_file', help='Merged data file')
    parser.add_argument('--mode', choices=['tensile', 'shear'], help='Test mode')
    parser.add_argument('--stress', help='Stress column name')
    parser.add_argument('--separation', help='Separation column name')
    parser.add_argument('--units', choices=['GPa', 'MPa'], default='GPa', help='Stress units')
    parser.add_argument('--smooth', action='store_true', default=True, help='Smooth data')
    parser.add_argument('--no-smooth', dest='smooth', action='store_false')
    parser.add_argument('--window', type=int, default=15, help='Smoothing window')
    parser.add_argument('--poly', type=int, default=3, help='Smoothing polynomial order')
    parser.add_argument('--stiffness-frac', type=float, default=0.05)
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--plot', nargs='?', const='czm_curve.png', help='Save plot')
    
    args = parser.parse_args()
    
    # Read data
    print(f"Reading: {args.input_file}")
    try:
        df = pd.read_csv(args.input_file, delim_whitespace=True)
    except:
        df = pd.read_csv(args.input_file)
    
    print(f"Data shape: {df.shape}")
    
    # Auto-detect or use specified
    if args.mode is None or args.stress is None or args.separation is None:
        mode, stress_col, sep_col = detect_mode_and_columns(df)
        mode = args.mode or mode
        stress_col = args.stress or stress_col
        sep_col = args.separation or sep_col
        print(f"Mode: {mode}, Stress: {stress_col}, Separation: {sep_col}")
    else:
        mode = args.mode
        stress_col = args.stress
        sep_col = args.separation
    
    if stress_col not in df.columns or sep_col not in df.columns:
        print(f"ERROR: Columns not found")
        sys.exit(1)
    
    # Extract data
    delta_A = df[sep_col].values
    T_raw = df[stress_col].values
    if args.units == 'GPa':
        T_raw = T_raw * 1000.0
    
    T_initial = T_raw[0]
    if T_initial < 0:
        print(f"Shifting stress by {abs(T_initial):.3f} MPa (initial stress was negative)")
        T_raw = T_raw - T_initial
    
    # Extract mode identifier
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    mode_match = re.search(r'(xy|xz|yz|[xyz])', base_filename, re.IGNORECASE)
    mode_identifier = mode_match.group(1).lower() if mode_match else 'data'
    
    # Calculate
    print(f"\nCalculating CZM parameters...")
    results = calculate_czm_parameters(
        delta_A, T_raw, 
        mode=mode,
        smooth=args.smooth,
        smooth_window=args.window,
        smooth_poly=args.poly,
        stiffness_fraction=args.stiffness_frac
    )
    
    print_results(results)
    
    # Save
    output_file = args.output or f'czm_parameters_{mode_identifier}.dat'
    save_parameters(results, output_file)
    
    # Plot
    if args.plot:
        if args.plot == 'czm_curve.png':
            plot_filename = f'czm_curve_{mode_identifier}.png'
        else:
            plot_filename = args.plot
        plot_traction_separation(results, save_plot=plot_filename)

if __name__ == '__main__':
    main()