# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:28:31 2025

@author: SOUMYA
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:06:20 2025

@author: DELL
"""

# ===========================================
# ROBUST COMBINED FJC, SAW, and WLC Model Analysis for P(x)
# With Enhanced Hybrid SAW Sampling and Comprehensive Diagnostics
# ===========================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, ks_2samp, truncnorm
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from numba import njit
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ===========================================
# Parameters
# ===========================================
a = 0.1  # Kuhn length (µm)
L = 2.0  # Confinement length (µm)
lp = 0.2  # Persistence length for WLC (µm)
x0 = 0.75  # Fixed tethering point (µm)

# Chain lengths to analyze (including N=60)
N_values = [5, 10, 15, 20, 25, 30, 40, 50, 60]

# Standardized parameters
N_BINS = 50
BOOTSTRAP_SAMPLES = 500  # Reduced from 1000

# ===========================================
# Truncated Gaussian Functions
# ===========================================
def truncated_gaussian(x, mu, sigma, L):
    """Truncated Gaussian distribution between 0 and L"""
    a = (0 - mu) / sigma
    b = (L - mu) / sigma
    return truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

def fit_truncated_gaussian(data, x_grid, weights=None, L=2.0):
    """Fit truncated Gaussian to data and return parameters"""
    if len(data) < 10:
        return np.nan, np.nan, np.zeros_like(x_grid)
    try:
        # Initial guess from regular Gaussian
        if weights is not None:
            mu0 = np.average(data, weights=weights)
            variance = np.average((data - mu0)**2, weights=weights)
            sigma0 = np.sqrt(variance)
        else:
            mu0 = np.mean(data)
            sigma0 = np.std(data)
        
        # Constrain parameters for reasonable fits
        bounds = ([0.1, 0.01], [L-0.1, L/2])
        
        # Fit truncated Gaussian
        if weights is not None:
            hist, bin_edges = np.histogram(data, bins=50, range=(0, L), density=True, weights=weights)
        else:
            hist, bin_edges = np.histogram(data, bins=50, range=(0, L), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Define wrapper function that includes L
        def truncated_gaussian_wrapper(x, mu, sigma):
            return truncated_gaussian(x, mu, sigma, L)
            
        popt, pcov = curve_fit(truncated_gaussian_wrapper, bin_centers, hist, 
                              p0=[mu0, sigma0], bounds=bounds, maxfev=5000)
        mu_fit, sigma_fit = popt
        fitted_pdf = truncated_gaussian(x_grid, mu_fit, sigma_fit, L)
        return mu_fit, sigma_fit, fitted_pdf
        
    except (RuntimeError, ValueError) as e:
        print(f"Truncated Gaussian fit failed for N={len(data)}: {e}")
        # Fallback to regular Gaussian if truncated fit fails
        if weights is not None:
            mu_fit = np.average(data, weights=weights)
            variance = np.average((data - mu_fit)**2, weights=weights)
            sigma_fit = np.sqrt(variance)
        else:
            mu_fit = np.mean(data)
            sigma_fit = np.std(data)
        # Create truncated version
        fitted_pdf = truncated_gaussian(x_grid, mu_fit, sigma_fit, L)
        return mu_fit, sigma_fit, fitted_pdf

# ===========================================
# FJC Gaussian Fitting Functions with Boundary Enforcement
# ===========================================
def fit_fjc_gaussian(fjc_data, x_grid, L=2.0):
    """Fit Gaussian to FJC data and return parameters with boundary enforcement"""
    if len(fjc_data) < 10:
        return np.nan, np.nan, np.zeros_like(x_grid)
    
    mean_fjc = np.mean(fjc_data)
    std_fjc = np.std(fjc_data)
    
    # Create FJC Gaussian PDF (not truncated)
    fjc_gaussian = norm.pdf(x_grid, loc=mean_fjc, scale=std_fjc)
    
    # Enforce P(x) = 0 at boundaries
    boundary_mask = (x_grid <= 0) | (x_grid >= L)
    fjc_gaussian[boundary_mask] = 0
    
    # Renormalize
    norm_factor = trapezoid(fjc_gaussian, x_grid)
    if norm_factor > 0:
        fjc_gaussian /= norm_factor
        
    return mean_fjc, std_fjc, fjc_gaussian

# ===========================================
# Bootstrap error estimation with ESS
# ===========================================
def bootstrap_errors(data, weights=None, n_bootstrap=500, alpha=0.05):  # Reduced from 1000
    """
    Calculate bootstrap confidence intervals with effective sample size
    """
    if len(data) < 10:
        return np.nan, np.nan, np.nan, np.nan, len(data)
    
    n = len(data)
    
    # Calculate effective sample size if weights are provided
    if weights is not None:
        ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    else:
        ess = n
        
    bootstrap_means = []
    bootstrap_stds = []
    
    for _ in range(n_bootstrap):
        if weights is not None:
            # Weighted bootstrap sampling
            indices = np.random.choice(n, size=n, replace=True, p=weights/np.sum(weights))
            sample = data[indices]
        else:
            sample = np.random.choice(data, size=n, replace=True)
            
        bootstrap_means.append(np.mean(sample))
        bootstrap_stds.append(np.std(sample))
    
    # Calculate confidence intervals
    mean_lower = np.percentile(bootstrap_means, 100*alpha/2)
    mean_upper = np.percentile(bootstrap_means, 100*(1-alpha/2))
    std_lower = np.percentile(bootstrap_stds, 100*alpha/2)
    std_upper = np.percentile(bootstrap_stds, 100*(1-alpha/2))
    
    mean_err = (mean_upper - mean_lower) / 2
    std_err = (std_upper - std_lower) / 2
    
    return mean_err, std_err, ess

def bootstrap_pdf_errors(data, x_grid, weights=None, n_bootstrap=50, alpha=0.05):  # Reduced from 100
    """
    Calculate bootstrap confidence intervals for PDF
    """
    if len(data) < 20:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)
    
    n = len(data)
    bootstrap_pdfs = []
    
    for _ in range(n_bootstrap):
        if weights is not None:
            indices = np.random.choice(n, size=n, replace=True, p=weights/np.sum(weights))
            sample = data[indices]
        else:
            sample = np.random.choice(data, size=n, replace=True)
            
        if len(sample) > 10:
            kde = gaussian_kde(sample)
            pdf = kde(x_grid)
            pdf = np.clip(pdf, 0, None)
            
            # Enforce P(x) = 0 at boundaries
            boundary_mask = (x_grid <= 0) | (x_grid >= L)
            pdf[boundary_mask] = 0
            
            # Normalize
            mask = (x_grid>=0)&(x_grid<=L)
            norm_factor = trapezoid(pdf[mask], x_grid[mask])
            if norm_factor > 0:
                pdf /= norm_factor
                
            bootstrap_pdfs.append(pdf)
    
    if len(bootstrap_pdfs) == 0:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)
        
    bootstrap_pdfs = np.array(bootstrap_pdfs)
    pdf_mean = np.mean(bootstrap_pdfs, axis=0)
    pdf_upper = np.percentile(bootstrap_pdfs, 100*(1-alpha/2), axis=0)
    pdf_lower = np.percentile(bootstrap_pdfs, 100*alpha/2, axis=0)  # FIXED: removed extra parenthesis
    
    return pdf_mean, (pdf_upper - pdf_lower) / 2

# ===========================================
# Analytical Fourier solution with Boundary Enforcement
# ===========================================
def analytical_Px_fourier(x_vals, x0, a, N, L, n_terms=500, decay_tol=1e-10):  # Reduced from 1000
    D = a**2 / 2
    G = np.zeros_like(x_vals)
    
    for n in range(1, n_terms):
        decay = np.exp(-(n * np.pi / L)**2 * D * N)
        if decay < decay_tol:
            break
        G += np.sin(n*np.pi*x0/L) * np.sin(n*np.pi*x_vals/L) * decay
        
    G *= (2 / L)
    
    # Enforce P(x) = 0 at boundaries
    boundary_mask = (x_vals <= 0) | (x_vals >= L)
    G[boundary_mask] = 0
    
    norm_factor = trapezoid(G, x_vals)
    return G / norm_factor if norm_factor > 0 else np.ones_like(x_vals)/L

# ===========================================
# FJC Monte Carlo
# ===========================================
def fjc_monte_carlo_Px(x0, a, N, L, n_walkers=100000, seed=42):  # Reduced from 200000
    np.random.seed(seed)
    final_positions = []
    
    for _ in range(n_walkers):
        x = x0
        alive = True
        for _ in range(N):
            x += a * np.random.randn()
            if x < 0 or x > L:  # absorbing boundary
                alive = False
                break
        if alive:
            final_positions.append(x)
            
    return np.array(final_positions), n_walkers, len(final_positions)

# ===========================================
# SAW UTILITY FUNCTIONS
# ===========================================
@njit
def generate_saw_step(current_pos, existing_chain, a, L, bead_radius, max_trials=20):
    """
    Generate a step for SAW with multiple trials to avoid overlaps
    """
    for trial in range(max_trials):
        # Generate random direction
        step = np.random.normal(0.0, 1.0, 3)
        step_norm = np.sqrt(np.sum(step**2))
        if step_norm > 1e-8:
            step /= step_norm
        else:
            continue
            
        new_pos = current_pos + a * step
        
        # Check boundary
        if new_pos[0] < 0 or new_pos[0] > L:
            continue
            
        # Check overlap with all previous beads
        overlap = False
        for i in range(len(existing_chain)):
            if np.linalg.norm(new_pos - existing_chain[i]) < 2 * bead_radius:
                overlap = True
                break
                
        if not overlap:
            return new_pos, True
            
    return current_pos, False  # Failed to find valid step

@njit
def generate_saw_perm_step(current_pos, existing_chain, a, L, bead_radius, k_trials=10):
    """
    Generate k trial steps and count valid ones for PERM weight calculation
    """
    valid_steps = []
    for _ in range(k_trials):
        step = np.random.normal(0.0, 1.0, 3)
        step_norm = np.sqrt(np.sum(step**2))
        if step_norm > 1e-8:
            step /= step_norm
        else:
            continue
            
        new_pos = current_pos + a * step
        
        # Check boundary
        if new_pos[0] < 0 or new_pos[0] > L:
            continue
            
        # Check overlap with all previous beads
        overlap = False
        for i in range(len(existing_chain)):
            if np.linalg.norm(new_pos - existing_chain[i]) < 2 * bead_radius:
                overlap = True
                break
                
        if not overlap:
            valid_steps.append(new_pos)
            
    return valid_steps

@njit
def check_self_avoidance(coords, new_segment, pivot_index, bead_radius):
    """
    Check if the pivoted segment avoids self-intersection
    """
    for i in range(len(new_segment)):
        for j in range(pivot_index):  # Check against original part
            if np.linalg.norm(new_segment[i] - coords[j]) < 2 * bead_radius:
                return False
        # Check within new segment
        for j in range(i+1, len(new_segment)):
            if np.linalg.norm(new_segment[i] - new_segment[j]) < 2 * bead_radius:
                return False
    return True

@njit
def check_confinement(coords, L):
    """
    Check if all beads are within confinement [0, L] in x-direction
    """
    for i in range(len(coords)):
        if coords[i, 0] < 0 or coords[i, 0] > L:
            return False
    return True

@njit
def rotate_vector_3d(vector, axis, angle):
    """
    Rotate a 3D vector around given axis by angle (radians)
    Using Rodrigues' rotation formula
    """
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodrigues' rotation formula
    rotated = (vector * cos_angle +
              np.cross(axis, vector) * sin_angle +
              axis * np.dot(axis, vector) * (1 - cos_angle))
    return rotated

@njit
def generate_pivot_move(coords, pivot_index, max_angle=np.pi):
    """
    Generate a pivot move by rotating the tail of the chain
    """
    # Choose random rotation axis and angle
    axis = np.random.normal(0, 1, 3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-8:
        axis /= axis_norm
    else:
        axis = np.array([1.0, 0.0, 0.0])
        
    angle = (np.random.random() - 0.5) * 2 * max_angle
    
    # Apply rotation to the tail
    pivot_point = coords[pivot_index]
    new_coords = coords.copy()
    
    for i in range(pivot_index + 1, len(coords)):
        # Vector from pivot point to current bead
        vec = coords[i] - pivot_point
        # Rotate the vector
        rotated_vec = rotate_vector_3d(vec, axis, angle)
        # New position
        new_coords[i] = pivot_point + rotated_vec
        
    return new_coords, pivot_index

@njit
def generate_initial_saw_chain(x0, a, N, L, bead_radius, max_attempts=5000):  # Reduced from 10000
    """
    Generate initial SAW chain for pivot algorithm using simple growth
    """
    coords = np.zeros((N+1, 3))
    coords[0] = [x0, 0.0, 0.0]
    
    for i in range(1, N+1):
        found_step = False
        for attempt in range(max_attempts):
            # Random direction
            step = np.random.normal(0, 1, 3)
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-8:
                step = step / step_norm * a
            else:
                continue
                
            new_pos = coords[i-1] + step
            
            # Check boundary
            if new_pos[0] < 0 or new_pos[0] > L:
                continue
                
            # Check self-avoidance
            collision = False
            for j in range(i):
                if np.linalg.norm(new_pos - coords[j]) < 2 * bead_radius:
                    collision = True
                    break
                    
            if not collision:
                coords[i] = new_pos
                found_step = True
                break
                
        if not found_step:
            return None
            
    return coords

# ===========================================
# IMPROVED ADAPTIVE SAW SAMPLING FUNCTIONS
# ===========================================
def improved_adaptive_saw_monte_carlo_Px(x0, a, N, L, bead_radius=0.03, n_walkers=2000,
                                        pivot_threshold=40, perm_threshold=30, max_attempts_factor=1000):
    """
    Improved Adaptive SAW Sampler with enhanced sampling strategies for different chain length regimes
    Maintains same interface and output structure as original function
    """
    
    # ADAPTIVE SAMPLE SIZING BASED ON CHAIN LENGTH
    # Longer chains require more samples for reliable statistics
    if N >= 50:
        effective_walkers = 3500  # Increased sampling for very long chains
        print(f"Using ADAPTIVE sampling: N={N} -> {effective_walkers} walkers")
    elif N >= 40:
        effective_walkers = 3000  # Moderate increase for long chains
        print(f"Using ADAPTIVE sampling: N={N} -> {effective_walkers} walkers")
    elif N >= 30:
        effective_walkers = 2500  # Slight increase for transition region
        print(f"Using ADAPTIVE sampling: N={N} -> {effective_walkers} walkers")
    else:
        effective_walkers = n_walkers  # Keep original for short chains
        print(f"Using STANDARD sampling: N={N} -> {effective_walkers} walkers")

    # ENHANCED HYBRID STRATEGY WITH IMPROVED PARAMETERS
    if N >= pivot_threshold:
        print(f"Using IMPROVED PIVOT algorithm for SAW N={N}")
        return improved_pivot_saw(x0, a, N, L, bead_radius, n_samples=effective_walkers)
    elif N >= perm_threshold:
        print(f"Using IMPROVED PERM algorithm for SAW N={N}")
        return improved_perm_saw(x0, a, N, L, bead_radius, n_walkers=effective_walkers)
    else:
        print(f"Using IMPROVED Simple MC for SAW N={N}")
        return improved_simple_saw(x0, a, N, L, bead_radius, n_walkers=effective_walkers)

def improved_simple_saw(x0, a, N, L, bead_radius=0.03, n_walkers=2000, max_attempts_factor=1000):
    """
    Improved Simple SAW Monte Carlo with better chain growth and diagnostics
    """
    @njit
    def generate_improved_saw_chain(N, a, x0, L, bead_radius, max_trials=25):  # Increased trials
        coords = np.zeros((N+1, 3))
        coords[0, 0] = x0
        
        for i in range(1, N+1):
            best_pos = None
            best_clearance = -1.0
            found_valid = False
            
            # Try multiple directions and pick the one with best clearance
            for trial in range(max_trials):
                step = np.random.normal(0.0, 1.0, 3)
                step_norm = np.sqrt(np.sum(step**2))
                if step_norm > 1e-8:
                    step /= step_norm
                else:
                    continue
                    
                new_pos = coords[i-1] + a * step
                
                # Check boundary
                if new_pos[0] < 0 or new_pos[0] > L:
                    continue
                    
                # Calculate minimum distance to existing chain (clearance)
                min_dist = float('inf')
                for j in range(i):
                    dist = np.linalg.norm(new_pos - coords[j])
                    if dist < min_dist:
                        min_dist = dist
                
                # Prefer positions with larger clearance from existing chain
                if min_dist >= 2 * bead_radius:
                    if min_dist > best_clearance:
                        best_clearance = min_dist
                        best_pos = new_pos.copy()
                    found_valid = True
            
            if found_valid and best_pos is not None:
                coords[i] = best_pos
            else:
                return -1.0  # Chain got stuck
                
        return coords[-1, 0]
    
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * n_walkers * (N//10 + 1)
    
    # Progress tracking
    progress_interval = max(500, n_walkers // 10)
    
    while len(ends) < n_walkers and attempts < max_attempts:
        val = generate_improved_saw_chain(N, a, x0, L, bead_radius)
        if val >= 0:
            ends.append(val)
        attempts += 1
        
        # Progress reporting for long runs
        if attempts % progress_interval == 0 and len(ends) > 0:
            current_acceptance = len(ends) / attempts
            print(f"Improved Simple SAW N={N}: Progress {len(ends)}/{n_walkers}, "
                  f"acceptance: {current_acceptance:.4f}")
    
    acceptance_rate = len(ends) / attempts if attempts > 0 else 0
    ess = len(ends)
    
    if len(ends) < n_walkers:
        print(f"⚠️ Improved Simple SAW at N={N}: {len(ends)} of {n_walkers} accepted, "
              f"acceptance rate: {acceptance_rate:.6f}, ESS: {ess}")
    else:
        print(f"✅ Improved Simple SAW at N={N}: All {n_walkers} samples accepted, "
              f"final acceptance: {acceptance_rate:.6f}")
              
    return np.array(ends), None, attempts, len(ends), ess

def improved_perm_saw(x0, a, N, L, bead_radius=0.03, n_walkers=2000,
                     k_trials=20, c_min=0.2, c_max=4.0, population_limit_factor=4):
    """
    Improved PERM algorithm with better weight management and population control
    """
    
    # Adaptive parameters based on chain length
    if N >= 35:
        k_trials = 25  # More trials for longer chains in PERM regime
        c_min = 0.15   # More lenient pruning
        print(f"Improved PERM SAW N={N}: Using extended-chain parameters")
        
    population = []
    for i in range(n_walkers):
        chain = [np.array([x0, 0.0, 0.0])]
        weight = 1.0
        population.append((chain, weight, i))
        
    current_step = 1
    max_population = population_limit_factor * n_walkers
    
    print(f"Improved PERM SAW N={N}: Starting with {len(population)} walkers, "
          f"k_trials={k_trials}, c_min={c_min}, c_max={c_max}")
          
    while current_step <= N and len(population) > 0:
        new_population = []
        total_weight = 0.0
        trapped_chains = 0
        
        for chain, weight, walker_id in population:
            current_pos = chain[-1]
            valid_steps = generate_saw_perm_step(current_pos, chain, a, L, bead_radius, k_trials)
            
            if len(valid_steps) == 0:
                trapped_chains += 1
                continue
                
            step_weight = weight * len(valid_steps) / k_trials
            
            # Choose step with bias toward larger clearance
            if len(valid_steps) > 1:
                # Calculate clearances and choose step with best clearance
                clearances = []
                for step in valid_steps:
                    min_dist = float('inf')
                    for existing_pos in chain:
                        dist = np.linalg.norm(step - existing_pos)
                        if dist < min_dist:
                            min_dist = dist
                    clearances.append(min_dist)
                
                # Weighted selection based on clearance
                clearances = np.array(clearances)
                probs = clearances / np.sum(clearances)
                chosen_idx = np.random.choice(len(valid_steps), p=probs)
                chosen_step = valid_steps[chosen_idx]
            else:
                chosen_step = valid_steps[0]
                
            new_chain = chain + [chosen_step]
            new_population.append((new_chain, step_weight, walker_id))
            total_weight += step_weight
            
        if len(new_population) == 0:
            print(f"⚠️ Improved PERM SAW N={N}: All chains trapped at step {current_step}")
            break
            
        avg_weight = total_weight / len(new_population)
        final_population = []
        
        # Enhanced pruning/enrichment with progressive tightening
        progressive_c_min = c_min * (1 + 0.1 * (current_step / N))  # Tighten as chain grows
        progressive_c_max = c_max * (1 - 0.05 * (current_step / N))  # Tighten as chain grows
        
        for chain, weight, walker_id in new_population:
            ratio = weight / avg_weight if avg_weight > 0 else 0
            
            if ratio < progressive_c_min:
                survival_prob = ratio / progressive_c_min
                if np.random.random() < survival_prob:
                    final_population.append((chain, avg_weight * survival_prob, walker_id))
            elif ratio > progressive_c_max:
                num_copies = min(int(np.sqrt(ratio)) + 1, 5)  # Controlled growth
                for i in range(num_copies):
                    final_population.append((chain, weight / num_copies, walker_id * 1000 + i))
            else:
                final_population.append((chain, weight, walker_id))
                
        # Population control with weight-based sampling
        if len(final_population) > max_population:
            weights = [w for _, w, _ in final_population]
            total_w = sum(weights)
            if total_w > 0:
                probs = [w / total_w for w in weights]
                indices = np.random.choice(len(final_population), size=max_population, replace=False, p=probs)
                final_population = [final_population[i] for i in indices]
            else:
                indices = np.random.choice(len(final_population), size=max_population, replace=False)
                final_population = [final_population[i] for i in indices]
                
        population = final_population
        current_step += 1
        
        if current_step % 5 == 0:  # More frequent reporting for PERM
            current_acceptance = len(population) / n_walkers
            print(f"Improved PERM SAW N={N}: Step {current_step}, population: {len(population)}, "
                  f"trapped: {trapped_chains}, current acceptance: {current_acceptance:.4f}")
                  
    # Extract final results
    final_positions = []
    final_weights = []
    total_final_weight = 0.0
    complete_chains = 0
    
    for chain, weight, _ in population:
        if len(chain) == N + 1:
            final_positions.append(chain[-1][0])
            final_weights.append(weight)
            total_final_weight += weight
            complete_chains += 1
            
    final_positions = np.array(final_positions)
    final_weights = np.array(final_weights)
    
    if total_final_weight > 0:
        final_weights /= total_final_weight
        
    acceptance_rate = complete_chains / n_walkers
    ess = (np.sum(final_weights) ** 2) / np.sum(final_weights ** 2) if len(final_weights) > 0 else 0
    
    print(f"Improved PERM SAW N={N}: {complete_chains} complete chains, "
          f"acceptance: {acceptance_rate:.4f}, ESS: {ess:.1f}")
          
    return final_positions, final_weights, n_walkers, complete_chains, ess

def improved_pivot_saw(x0, a, N, L, bead_radius=0.03, n_samples=2000,
                      n_equilibration=2500, max_angle=np.pi/2, pivot_attempts_factor=8):  # Increased attempts
    """
    Improved pivot algorithm with better equilibration and sampling
    """
    # Generate initial chain with more attempts for reliability
    print(f"Improved Pivot SAW N={N}: Generating initial chain...")
    initial_chain = generate_initial_saw_chain(x0, a, N, L, bead_radius, max_attempts=20000)
    
    if initial_chain is None:
        print(f"❌ Improved Pivot SAW N={N}: Failed to generate initial chain")
        # Fallback: try to generate any valid chain with reduced requirements
        print("Attempting fallback chain generation...")
        initial_chain = generate_initial_saw_chain(x0, a, N, L, bead_radius*0.9, max_attempts=10000)  # Relaxed radius
        if initial_chain is None:
            return np.array([]), None, 0, 0, 0
        
    coords = initial_chain
    accepted = 0
    attempts = 0
    
    print(f"Improved Pivot SAW N={N}: Starting pivot moves (equilibration: {n_equilibration})")
    
    # Enhanced equilibration with acceptance rate monitoring
    recent_acceptances = []
    window_size = 100
    
    for step in range(n_equilibration):
        if len(coords) <= 2:
            break
            
        pivot_accepted = False
        for pivot_attempt in range(pivot_attempts_factor):
            pivot_index = np.random.randint(1, len(coords) - 1)
            new_coords, pivot_idx = generate_pivot_move(coords, pivot_index, max_angle)
            attempts += 1
            
            if (check_self_avoidance(coords, new_coords[pivot_index:], pivot_index, bead_radius) and
                check_confinement(new_coords, L)):
                coords = new_coords
                accepted += 1
                pivot_accepted = True
                break
                
        # Monitor recent acceptance rate
        recent_acceptances.append(1 if pivot_accepted else 0)
        if len(recent_acceptances) > window_size:
            recent_acceptances.pop(0)
            
        # Adaptive angle adjustment based on recent acceptance
        if step % 200 == 0 and len(recent_acceptances) == window_size:
            recent_rate = np.mean(recent_acceptances)
            if recent_rate < 0.1:
                max_angle *= 0.8  # Reduce angle for better acceptance
                print(f"Improved Pivot SAW N={N}: Reduced max_angle to {max_angle:.3f}")
            elif recent_rate > 0.3:
                max_angle = min(max_angle * 1.2, np.pi)  # Increase angle for larger moves
                print(f"Improved Pivot SAW N={N}: Increased max_angle to {max_angle:.3f}")
                
        if (step + 1) % 1000 == 0:
            current_acceptance = accepted / attempts if attempts > 0 else 0
            print(f"Improved Pivot SAW N={N}: Equilibration {step+1}/{n_equilibration}, "
                  f"acceptance: {current_acceptance:.4f}")
                  
    equilibration_acceptance = accepted / attempts if attempts > 0 else 0
    print(f"Improved Pivot SAW N={N}: Equilibration complete, acceptance: {equilibration_acceptance:.4f}")
    
    # Production phase with guaranteed sample count
    accepted_prod = 0
    attempts_prod = 0
    final_positions = []
    
    # Adaptive sampling interval based on chain length and acceptance rate
    base_interval = max(3, min(10, 30 // (N // 15 + 1)))
    
    # Ensure we collect exactly n_samples
    samples_collected = 0
    total_production_steps = n_samples * base_interval * 2  # Generous upper bound
    
    for step in range(total_production_steps):
        if samples_collected >= n_samples:
            break
            
        if len(coords) <= 2:
            break
            
        pivot_accepted = False
        for pivot_attempt in range(pivot_attempts_factor):
            pivot_index = np.random.randint(1, len(coords) - 1)
            new_coords, pivot_idx = generate_pivot_move(coords, pivot_index, max_angle)
            attempts_prod += 1
            
            if (check_self_avoidance(coords, new_coords[pivot_index:], pivot_index, bead_radius) and
                check_confinement(new_coords, L)):
                coords = new_coords
                accepted_prod += 1
                pivot_accepted = True
                break
                
        # Store sample at adaptive intervals
        if step % base_interval == 0 and pivot_accepted:
            final_positions.append(coords[-1, 0])
            samples_collected += 1
            
    # If we didn't get enough samples, continue until we do
    extra_steps = 0
    while samples_collected < n_samples and extra_steps < n_samples * 10:
        if len(coords) <= 2:
            break
            
        pivot_accepted = False
        for pivot_attempt in range(pivot_attempts_factor):
            pivot_index = np.random.randint(1, len(coords) - 1)
            new_coords, pivot_idx = generate_pivot_move(coords, pivot_index, max_angle)
            attempts_prod += 1
            
            if (check_self_avoidance(coords, new_coords[pivot_index:], pivot_index, bead_radius) and
                check_confinement(new_coords, L)):
                coords = new_coords
                accepted_prod += 1
                pivot_accepted = True
                break
                
        if pivot_accepted:
            final_positions.append(coords[-1, 0])
            samples_collected += 1
            
        extra_steps += 1
        
    acceptance_rate_prod = accepted_prod / attempts_prod if attempts_prod > 0 else 0
    total_acceptance = (accepted + accepted_prod) / (attempts + attempts_prod)
    
    print(f"Improved Pivot SAW N={N}: Generated {len(final_positions)} samples, "
          f"production acceptance: {acceptance_rate_prod:.4f}, "
          f"total acceptance: {total_acceptance:.4f}")
          
    return (np.array(final_positions), None, attempts + attempts_prod,
            len(final_positions), len(final_positions))

# ===========================================
# ENHANCED WLC WITH PERSISTENCE LENGTH OPTIMIZATION
# ===========================================
@njit
def generate_wlc_chain(N, a, lp, x0, L, temperature_kT=1.0):
    # Initialize chain coordinates and directions
    coords = np.zeros((N+1, 3))
    tangents = np.zeros((N, 3))
    
    # Start at x0, random initial direction
    coords[0, 0] = x0
    initial_dir = np.random.normal(0, 1, 3)
    initial_dir /= np.sqrt(np.sum(initial_dir**2))
    tangents[0] = initial_dir
    
    # Generate the chain
    for i in range(1, N+1):
        # Propose new direction based on bending energy
        current_tangent = tangents[i-1]
        
        # Generate trial direction with bias toward current direction
        bending_stiffness = lp / a  # dimensionless bending parameter
        
        # Sample from von Mises-Fisher distribution (approximation)
        kappa = bending_stiffness  # concentration parameter
        
        # Marsaglia method for von Mises-Fisher sampling
        while True:
            v = np.random.normal(0, 1, 3)
            v_norm = np.sqrt(np.sum(v**2))
            if v_norm > 1e-8:
                v /= v_norm
                break
                
        # Bias toward current direction
        if kappa > 0:
            w = np.random.rand()
            w0 = (1.0 - np.exp(-2.0 * kappa)) * w + np.exp(-2.0 * kappa)
            cos_theta = 1.0 + np.log(w0) / kappa
            
            # Ensure cos_theta is within valid range
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            
            # Generate perpendicular component
            perpendicular = v - np.dot(v, current_tangent) * current_tangent
            perp_norm = np.sqrt(np.sum(perpendicular**2))
            if perp_norm > 1e-8:
                perpendicular /= perp_norm
                
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            new_direction = cos_theta * current_tangent + sin_theta * perpendicular
        else:
            new_direction = v  # random direction for kappa=0
            
        new_direction /= np.sqrt(np.sum(new_direction**2))
        
        # Update position
        coords[i] = coords[i-1] + a * new_direction
        
        if i < N:
            tangents[i] = new_direction
            
        # Check confinement
        if coords[i, 0] < 0 or coords[i, 0] > L:
            return -1.0
            
    return coords[-1, 0]

def enhanced_wlc_monte_carlo_Px(x0, a, N, L, lp=0.2, temperature_kT=1.0,
                               n_walkers=30000, max_attempts_factor=150):  # Reduced
    """
    Enhanced WLC Monte Carlo with persistence length optimization and better diagnostics
    """
    # Adaptive persistence length based on chain length
    if N <= 15:
        enhanced_lp = lp * 1.2  # Slightly stiffer for short chains
    elif N >= 40:
        enhanced_lp = lp * 0.9  # Slightly more flexible for long chains
    else:
        enhanced_lp = lp
        
    if enhanced_lp != lp:
        print(f"Enhanced WLC N={N}: Using adaptive lp={enhanced_lp:.3f} (was {lp:.3f})")
    else:
        print(f"Enhanced WLC N={N}: Using lp={lp:.3f}")
        
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * n_walkers
    
    # Progress tracking
    progress_interval = max(2000, n_walkers // 5)  # Less frequent
    
    while len(ends) < n_walkers and attempts < max_attempts:
        val = generate_wlc_chain(N, a, enhanced_lp, x0, L, temperature_kT)
        if val >= 0:
            ends.append(val)
        attempts += 1
        
        # Progress reporting
        if attempts % progress_interval == 0:
            current_acceptance = len(ends) / attempts if attempts > 0 else 0
            print(f"Enhanced WLC N={N}: Progress: {len(ends)}/{n_walkers}, "
                  f"attempts: {attempts}, current acceptance: {current_acceptance:.4f}")
                  
    acceptance_rate = len(ends) / attempts if attempts > 0 else 0
    ess = len(ends)
    
    if len(ends) < n_walkers:
        print(f"⚠️ Enhanced WLC at N={N}: {len(ends)} of {n_walkers} accepted, "
              f"acceptance rate: {acceptance_rate:.6f}")
    else:
        print(f"✅ Enhanced WLC at N={N}: All {n_walkers} samples accepted, "
              f"final acceptance: {acceptance_rate:.6f}")
              
    return np.array(ends), None, attempts, len(ends), ess

# ===========================================
# IMPROVED KDE for smooth distribution with Boundary Enforcement
# ===========================================
def improved_absorbing_kde(data, x_vals, weights=None, bw_factor=1.3, L=2.0, smoothing_sigma=1.0):
    """
    Improved KDE with better smoothing for SAW curves
    """
    if len(data) < 20:
        return None
        
    try:
        if weights is not None and np.sum(weights) > 0:
            # Weighted KDE
            kde = gaussian_kde(data, weights=weights)
        else:
            kde = gaussian_kde(data)
            
        # Use slightly larger bandwidth for smoother curves
        kde.set_bandwidth(bw_method=kde.factor * bw_factor)
        pdf = kde(x_vals)
        pdf = np.clip(pdf, 0, None)
        
        # Enforce P(x) = 0 at boundaries with smoother transition
        boundary_width = 0.1
        left_boundary = np.exp(-(x_vals - 0)**2 / (2 * boundary_width**2))
        right_boundary = np.exp(-(x_vals - L)**2 / (2 * boundary_width**2))
        boundary_mask = np.clip(left_boundary + right_boundary, 0, 1)
        pdf *= (1 - boundary_mask)
        
        # Apply Gaussian smoothing to reduce unevenness
        if smoothing_sigma > 0:
            pdf = gaussian_filter1d(pdf, sigma=smoothing_sigma)
            
        # Normalize
        mask = (x_vals>=0)&(x_vals<=L)
        norm_factor = trapezoid(pdf[mask], x_vals[mask])
        if norm_factor > 0:
            pdf /= norm_factor
            
        return pdf
        
    except Exception as e:
        print(f"KDE failed: {e}")
        return None

# ===========================================
# ENHANCED FIT METRICS WITH IMPROVED ROBUSTNESS
# ===========================================
def enhanced_calculate_fit_metrics(data, expected_pdf, x_grid, weights=None, L=2.0,
                                  min_samples_per_bin=5, confidence_level=0.95):
    """
    Enhanced fit metrics with improved robustness and confidence intervals
    """
    if len(data) < 30:  # More conservative threshold
        return {
            'KL': np.nan, 'KS_stat': np.nan, 'KS_pval': np.nan,
            'Chi2': np.nan, 'Reduced_Chi2': np.nan, 'DOF': np.nan,
            'MeanDiff': np.nan, 'StdRatio': np.nan,
            'Valid_Chi2': False, 'Chi2_Warning': f'Insufficient data: {len(data)} < 30',
            'Confidence_Level': confidence_level
        }
        
    # Enhanced adaptive binning
    n_bins = max(8, min(40, len(data) // min_samples_per_bin))
    if weights is not None:
        effective_samples = 1.0 / np.sum(weights**2) if np.sum(weights) > 0 else len(data)
        n_bins = max(8, min(30, int(effective_samples // min_samples_per_bin)))
        
    bin_edges = np.linspace(0, L, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Enhanced histogram calculation
    if weights is not None:
        hist_obs, _ = np.histogram(data, bins=bin_edges, weights=weights, density=True)
        counts_obs, _ = np.histogram(data, bins=bin_edges, weights=weights)
        total_weight = np.sum(weights)
        effective_n = (np.sum(weights)**2) / np.sum(weights**2)  # Effective sample size
    else:
        hist_obs, _ = np.histogram(data, bins=bin_edges, density=True)
        counts_obs, _ = np.histogram(data, bins=bin_edges)
        total_weight = len(data)
        effective_n = len(data)
        
    expected_vals = np.interp(bin_centers, x_grid, expected_pdf)
    
    # Enhanced probability calculation with smoothing
    p_obs = hist_obs * bin_width
    p_exp = expected_vals * bin_width
    
    p_obs_sum = np.sum(p_obs)
    p_exp_sum = np.sum(p_exp)
    
    if p_obs_sum > 0:
        p_obs /= p_obs_sum
    if p_exp_sum > 0:
        p_exp /= p_exp_sum
        
    # Enhanced regularization
    eps = 1e-10
    smoothing = 1e-8 * np.maximum(np.max(p_obs), np.max(p_exp))
    p_obs = np.clip(p_obs, eps + smoothing, None)
    p_exp = np.clip(p_exp, eps + smoothing, None)
    
    # KL divergence with enhanced stability
    KL = np.sum(p_obs * np.log(p_obs / p_exp))
    
    # Enhanced KS test
    if weights is None:
        KS_stat, KS_pval = ks_2samp(data,
                                   np.random.choice(x_grid, size=min(len(data), 5000),
                                                   p=expected_pdf/np.sum(expected_pdf)))
    else:
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        ecdf_obs = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        ecdf_exp = np.interp(sorted_data, x_grid, np.cumsum(expected_pdf) * (x_grid[1]-x_grid[0]))
        KS_stat = np.max(np.abs(ecdf_obs - ecdf_exp))
        KS_pval = np.nan  # p-value calculation for weighted KS is complex
        
    # Enhanced Chi-square test with better validation
    expected_counts = expected_vals * total_weight * bin_width
    
    # More robust bin validation
    mask = (expected_counts > 5) & (counts_obs > 2) & (p_obs > 1e-6) & (p_exp > 1e-6)
    valid_bins = np.sum(mask)
    
    if valid_bins < 8:  # More conservative requirement
        Chi2 = np.nan
        Reduced_Chi2 = np.nan
        DOF = np.nan
        Valid_Chi2 = False
        Chi2_Warning = f"Only {valid_bins} valid bins (need ≥8)"
    else:
        Chi2 = np.sum((counts_obs[mask] - expected_counts[mask])**2 / expected_counts[mask])
        DOF = valid_bins - 1
        Reduced_Chi2 = Chi2 / DOF if DOF > 0 else np.nan
        Valid_Chi2 = True
        Chi2_Warning = None
        
    # Enhanced mean and std calculation
    if weights is not None:
        mean_data = np.average(data, weights=weights)
        # Weighted standard deviation with Bessel's correction
        variance = np.average((data - mean_data)**2, weights=weights)
        std_data = np.sqrt(variance * effective_n / (effective_n - 1)) if effective_n > 1 else np.sqrt(variance)
    else:
        mean_data = np.mean(data)
        std_data = np.std(data, ddof=1)
        
    # Enhanced analytic moments calculation
    dx = x_grid[1] - x_grid[0]
    mean_analytic = np.sum(x_grid * expected_pdf) * dx
    std_analytic = np.sqrt(np.sum((x_grid - mean_analytic)**2 * expected_pdf) * dx)
    
    mean_diff = mean_data - mean_analytic
    std_ratio = std_data / std_analytic if std_analytic > 0 else np.nan
    
    return {
        'KL': KL, 'KS_stat': KS_stat, 'KS_pval': KS_pval,
        'Chi2': Chi2, 'Reduced_Chi2': Reduced_Chi2, 'DOF': DOF,
        'MeanDiff': mean_diff, 'StdRatio': std_ratio,
        'Valid_Chi2': Valid_Chi2, 'Chi2_Warning': Chi2_Warning,
        'Confidence_Level': confidence_level,
        'Effective_Samples': effective_n,
        'Valid_Bins': valid_bins
    }

# ===========================================
# PLOTTING FUNCTIONS FOR FIT METRICS
# ===========================================
def plot_fit_metrics_comparison(fourier_metrics_df, fjc_gaussian_metrics_df, truncated_gaussian_metrics_df,
                               save_path='fit_metrics_comparison.pdf'):
    """
    Create comparison plots for Fourier, FJC Gaussian, and Truncated Gaussian fit metrics
    """
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    # Replace Reduced_Chi2 with KS in metrics list
    metrics = ['Mean', 'MeanDiff', 'Std', 'StdRatio', 'KL', 'KS']
    y_labels = ['Mean (μm)', 'Mean Difference (μm)', 'Std Dev (μm)', 'Std Dev Ratio',
                'KL Divergence', 'KS Statistic']
    
    models = ['FJC', 'SAW', 'WLC']
    colors = ['blue', 'green', 'orange']
    markers = ['o', 's', '^']
    line_styles = ['-', '--', ':']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot Fourier fit metrics
        for j, model in enumerate(models):
            mean_values = []
            for N in N_values:
                row = fourier_metrics_df[(fourier_metrics_df['N'] == N) & 
                                        (fourier_metrics_df['Model'] == model)]
                if not row.empty:
                    if metric == 'Mean':
                        value = row[f'Mean_{model}'].values[0]
                    elif metric == 'MeanDiff':
                        value = row[f'MeanDiff_{model}'].values[0]
                    elif metric == 'Std':
                        value = row[f'Std_{model}'].values[0]
                    elif metric == 'StdRatio':
                        value = row[f'StdRatio_{model}'].values[0]
                    elif metric == 'KL':
                        value = row[f'KL_{model}'].values[0]
                    elif metric == 'KS':
                        value = row[f'KS_{model}'].values[0]
                    mean_values.append(value)
                else:
                    mean_values.append(np.nan)
                    
            ax.plot(N_values, mean_values, marker=markers[j], color=colors[j],
                   linewidth=3, markersize=10, linestyle=line_styles[0],
                   label=f'{model} (Fourier)')
                   
        # Plot FJC Gaussian fit metrics
        for j, model in enumerate(models):
            mean_values = []
            for N in N_values:
                row = fjc_gaussian_metrics_df[(fjc_gaussian_metrics_df['N'] == N) & 
                                             (fjc_gaussian_metrics_df['Model'] == model)]
                if not row.empty:
                    if metric == 'Mean':
                        value = row[f'Mean_{model}'].values[0]
                    elif metric == 'MeanDiff':
                        value = row[f'MeanDiff_{model}'].values[0]
                    elif metric == 'Std':
                        value = row[f'Std_{model}'].values[0]
                    elif metric == 'StdRatio':
                        value = row[f'StdRatio_{model}'].values[0]
                    elif metric == 'KL':
                        value = row[f'KL_{model}'].values[0]
                    elif metric == 'KS':
                        value = row[f'KS_{model}'].values[0]
                    mean_values.append(value)
                else:
                    mean_values.append(np.nan)
                    
            ax.plot(N_values, mean_values, marker=markers[j], color=colors[j],
                   linewidth=3, markersize=10, linestyle=line_styles[1],
                   label=f'{model} (FJC Gaussian)')
                   
        # Plot Truncated Gaussian fit metrics
        for j, model in enumerate(models):
            mean_values = []
            for N in N_values:
                row = truncated_gaussian_metrics_df[(truncated_gaussian_metrics_df['N'] == N) & 
                                                   (truncated_gaussian_metrics_df['Model'] == model)]
                if not row.empty:
                    if metric == 'Mean':
                        value = row[f'Mean_{model}'].values[0]
                    elif metric == 'MeanDiff':
                        value = row[f'MeanDiff_{model}'].values[0]
                    elif metric == 'Std':
                        value = row[f'Std_{model}'].values[0]
                    elif metric == 'StdRatio':
                        value = row[f'StdRatio_{model}'].values[0]
                    elif metric == 'KL':
                        value = row[f'KL_{model}'].values[0]
                    elif metric == 'KS':
                        value = row[f'KS_{model}'].values[0]
                    mean_values.append(value)
                else:
                    mean_values.append(np.nan)
                    
            ax.plot(N_values, mean_values, marker=markers[j], color=colors[j],
                   linewidth=3, markersize=10, linestyle=line_styles[2],
                   label=f'{model} (Truncated Gaussian)')
                   
        ax.set_xlabel('Chain Length (N)', fontsize=18, fontweight='bold')
        ax.set_ylabel(y_labels[i], fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        # Special y-axis limits for certain metrics
        if metric == 'KL':
            ax.set_ylim(bottom=0)
        elif metric == 'KS':
            ax.set_ylim(bottom=0)
            
        # Add reference line at 0.05 (common significance threshold for KS)
        if metric == 'KS':
            ax.axhline(y=0.05, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Common threshold (0.05)')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# ===========================================
# ENHANCED MAIN ANALYSIS FUNCTION WITH IMPROVED VISUALIZATION
# ===========================================
def enhanced_analyze_combined_models(N_values, a=0.1, lp=0.2, x0=0.75, L=2.0,
                                    n_walkers_fjc=100000, n_walkers_saw=2000, n_walkers_wlc=30000):  # Reduced
    """
    Enhanced combined analysis with maximum robustness and improved diagnostics
    """
    n_plots = len(N_values)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure with improved size and layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows))
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    results = []
    fourier_fit_results = []
    fjc_gaussian_fit_results = []
    truncated_gaussian_fit_results = []
    
    x_grid = np.linspace(0, L, 500)  # Higher resolution grid
    
    print("\n" + "="*80)
    print("ENHANCED HYBRID SAW SAMPLING STRATEGY")
    print("="*80)
    print("N < 30: Improved Simple Monte Carlo")
    print("30 ≤ N < 40: Improved PERM (transition-optimized)")
    print("N ≥ 40: Improved Pivot Algorithm (adaptive sampling)")
    print("WLC: Adaptive persistence length optimization")
    print("Fit Metrics: Enhanced robustness and validation")
    print("BOUNDARY ENFORCEMENT: P(x) = 0 at x=0 and x=L")
    print("NEW: Adaptive sample sizing based on chain length")
    print("="*80)
    
    for idx, N in enumerate(N_values):
        ax = axes[idx]
        print(f"\nAnalyzing N = {N}...")
        
        # Run ENHANCED simulations
        print(f"  FJC: ", end="")
        fjc_data, fjc_attempts, fjc_accepted = fjc_monte_carlo_Px(x0, a, N, L, n_walkers=n_walkers_fjc)
        fjc_acceptance = fjc_accepted / fjc_attempts
        fjc_mean_err, fjc_std_err, fjc_ess = bootstrap_errors(fjc_data)
        print(f"{fjc_accepted}/{fjc_attempts} accepted ({fjc_acceptance:.4f}), ESS: {fjc_ess}")
        
        print(f"  SAW: ", end="")
        saw_data, saw_weights, saw_attempts, saw_accepted, saw_ess = improved_adaptive_saw_monte_carlo_Px(
            x0, a, N, L, n_walkers=n_walkers_saw)
        saw_acceptance = saw_accepted / saw_attempts if saw_attempts > 0 else saw_accepted / n_walkers_saw
        saw_mean_err, saw_std_err, saw_ess_calc = bootstrap_errors(saw_data, saw_weights)
        print(f"{saw_accepted} samples, acceptance: {saw_acceptance:.4f}, ESS: {saw_ess_calc:.1f}")
        
        print(f"  WLC: ", end="")
        wlc_data, wlc_weights, wlc_attempts, wlc_accepted, wlc_ess = enhanced_wlc_monte_carlo_Px(
            x0, a, N, L, lp=lp, n_walkers=n_walkers_wlc)
        wlc_acceptance = wlc_accepted / wlc_attempts
        wlc_mean_err, wlc_std_err, wlc_ess_calc = bootstrap_errors(wlc_data)
        print(f"{wlc_accepted}/{wlc_attempts} accepted ({wlc_acceptance:.4f}), ESS: {wlc_ess_calc}")
        
        # Analytical solution
        fourier_pdf = analytical_Px_fourier(x_grid, x0, a, N, L, n_terms=100)  # Reduced from 200
        
        # FJC Gaussian fit
        fjc_mean, fjc_std, fjc_gaussian = fit_fjc_gaussian(fjc_data, x_grid, L)
        
        # Truncated Gaussian fits for each model
        print(f"  Truncated Gaussian Fitting: ", end="")
        fjc_trunc_mu, fjc_trunc_sigma, fjc_trunc_gaussian = fit_truncated_gaussian(fjc_data, x_grid, L=L)
        saw_trunc_mu, saw_trunc_sigma, saw_trunc_gaussian = fit_truncated_gaussian(saw_data, x_grid, weights=saw_weights, L=L)
        wlc_trunc_mu, wlc_trunc_sigma, wlc_trunc_gaussian = fit_truncated_gaussian(wlc_data, x_grid, L=L)
        print(f"FJC(μ={fjc_trunc_mu:.3f},σ={fjc_trunc_sigma:.3f}), SAW(μ={saw_trunc_mu:.3f},σ={saw_trunc_sigma:.3f}), WLC(μ={wlc_trunc_mu:.3f},σ={wlc_trunc_sigma:.3f})")
        
        # Enhanced bootstrap PDF errors for SAW with more samples for better error estimation
        saw_pdf_mean, saw_pdf_err = bootstrap_pdf_errors(saw_data, x_grid, saw_weights, n_bootstrap=100)  # Increased for better error bars
        
        # IMPROVED VISUALIZATION WITH ENHANCED SAW ERROR BARS
        # Plot FJC histogram with reduced alpha for better visibility
        ax.hist(fjc_data, bins=60, range=(0,L), density=True, alpha=0.3,  # Reduced bins and alpha
               color='cornflowerblue', edgecolor='black', linewidth=0.2,
               label=f'FJC MC (n={len(fjc_data)})')
        
        # Use improved KDE for SAW with smoothing and error bands
        if len(saw_data) >= 40:
            saw_pdf_improved = improved_absorbing_kde(saw_data, x_grid, weights=saw_weights, L=L, smoothing_sigma=1.0)
            if saw_pdf_improved is not None:
                # Plot SAW with error bands
                if saw_pdf_mean is not None and saw_pdf_err is not None:
                    # Main SAW curve
                    ax.plot(x_grid, saw_pdf_improved, color='green', lw=3.5, label='SAW MC (KDE)')
                    # Error band
                    ax.fill_between(x_grid, saw_pdf_improved - saw_pdf_err, saw_pdf_improved + saw_pdf_err, 
                                   color='green', alpha=0.3, label='SAW MC bootstrap error')
                else:
                    ax.plot(x_grid, saw_pdf_improved, color='green', lw=3.5, label='SAW MC (KDE)')
        elif len(saw_data) > 15:
            n_bins_saw = max(8, min(25, len(saw_data)//12))
            ax.hist(saw_data, bins=n_bins_saw, range=(0,L), density=True, alpha=0.5,
                   color='green', edgecolor='black', linewidth=0.3,
                   label=f'SAW MC (n={len(saw_data)})')
        
        # Enhanced WLC visualization with improved KDE
        if len(wlc_data) >= 500:  # Reduced threshold
            wlc_pdf = improved_absorbing_kde(wlc_data, x_grid, L=L, bw_factor=1.1)
            if wlc_pdf is not None:
                ax.plot(x_grid, wlc_pdf, color='darkorange', lw=3.5, label='WLC MC (KDE)')
        else:
            n_bins_wlc = max(20, min(50, len(wlc_data)//20))
            ax.hist(wlc_data, bins=n_bins_wlc, range=(0,L), density=True, alpha=0.4,
                   color='darkorange', edgecolor='black', linewidth=0.3,
                   label=f'WLC MC (n={len(wlc_data)})')
        
        # Plot analytical solution with thicker line
        ax.plot(x_grid, fourier_pdf, 'k-', lw=3.5, label='Fourier analytic', alpha=0.9)
        
        # Plot FJC Gaussian fit (for all models) with thicker line
        if not np.isnan(fjc_mean):
            ax.plot(x_grid, fjc_gaussian, 'r--', lw=3.0,
                   label=f'FJC Gaussian (μ={fjc_mean:.3f}, σ={fjc_std:.3f})', alpha=0.8)
        
        # Plot Truncated Gaussian fits with thicker lines
        ax.plot(x_grid, fjc_trunc_gaussian, 'm:', lw=2.5,
               label=f'FJC Trunc Gaussian', alpha=0.8)
        ax.plot(x_grid, saw_trunc_gaussian, 'c:', lw=2.5,
               label=f'SAW Trunc Gaussian', alpha=0.8)
        ax.plot(x_grid, wlc_trunc_gaussian, 'y:', lw=2.5,
               label=f'WLC Trunc Gaussian', alpha=0.8)
        
        # Calculate ENHANCED fit metrics for Fourier solution
        fjc_metrics_fourier = enhanced_calculate_fit_metrics(fjc_data, fourier_pdf, x_grid, L=L)
        saw_metrics_fourier = enhanced_calculate_fit_metrics(saw_data, fourier_pdf, x_grid, saw_weights, L=L)
        wlc_metrics_fourier = enhanced_calculate_fit_metrics(wlc_data, fourier_pdf, x_grid, L=L)
        
        # Calculate ENHANCED fit metrics for FJC Gaussian
        fjc_metrics_gaussian = enhanced_calculate_fit_metrics(fjc_data, fjc_gaussian, x_grid, L=L)
        saw_metrics_gaussian = enhanced_calculate_fit_metrics(saw_data, fjc_gaussian, x_grid, saw_weights, L=L)
        wlc_metrics_gaussian = enhanced_calculate_fit_metrics(wlc_data, fjc_gaussian, x_grid, L=L)
        
        # Calculate ENHANCED fit metrics for Truncated Gaussian
        fjc_metrics_trunc = enhanced_calculate_fit_metrics(fjc_data, fjc_trunc_gaussian, x_grid, L=L)
        saw_metrics_trunc = enhanced_calculate_fit_metrics(saw_data, saw_trunc_gaussian, x_grid, saw_weights, L=L)
        wlc_metrics_trunc = enhanced_calculate_fit_metrics(wlc_data, wlc_trunc_gaussian, x_grid, L=L)
        
        # Store Fourier fit results
        fourier_fit_results.append({
            'N': N,
            'Model': 'FJC',
            'Mean_FJC': np.mean(fjc_data) if len(fjc_data) > 0 else np.nan,
            'MeanDiff_FJC': fjc_metrics_fourier['MeanDiff'],
            'Std_FJC': np.std(fjc_data) if len(fjc_data) > 0 else np.nan,
            'StdRatio_FJC': fjc_metrics_fourier['StdRatio'],
            'KL_FJC': fjc_metrics_fourier['KL'],
            'KS_FJC': fjc_metrics_fourier['KS_stat'],
            'Effective_Samples_FJC': fjc_metrics_fourier['Effective_Samples'],
            'Valid_Chi2_FJC': fjc_metrics_fourier['Valid_Chi2']
        })
        
        fourier_fit_results.append({
            'N': N,
            'Model': 'SAW',
            'Mean_SAW': np.average(saw_data, weights=saw_weights) if len(saw_data) > 0 and saw_weights is not None else np.mean(saw_data) if len(saw_data) > 0 else np.nan,
            'MeanDiff_SAW': saw_metrics_fourier['MeanDiff'],
            'Std_SAW': np.sqrt(np.average((saw_data - np.average(saw_data, weights=saw_weights))**2, weights=saw_weights)) if len(saw_data) > 0 and saw_weights is not None else np.std(saw_data) if len(saw_data) > 0 else np.nan,
            'StdRatio_SAW': saw_metrics_fourier['StdRatio'],
            'KL_SAW': saw_metrics_fourier['KL'],
            'KS_SAW': saw_metrics_fourier['KS_stat'],
            'Effective_Samples_SAW': saw_metrics_fourier['Effective_Samples'],
            'Valid_Chi2_SAW': saw_metrics_fourier['Valid_Chi2']
        })
        
        fourier_fit_results.append({
            'N': N,
            'Model': 'WLC',
            'Mean_WLC': np.mean(wlc_data) if len(wlc_data) > 0 else np.nan,
            'MeanDiff_WLC': wlc_metrics_fourier['MeanDiff'],
            'Std_WLC': np.std(wlc_data) if len(wlc_data) > 0 else np.nan,
            'StdRatio_WLC': wlc_metrics_fourier['StdRatio'],
            'KL_WLC': wlc_metrics_fourier['KL'],
            'KS_WLC': wlc_metrics_fourier['KS_stat'],
            'Effective_Samples_WLC': wlc_metrics_fourier['Effective_Samples'],
            'Valid_Chi2_WLC': wlc_metrics_fourier['Valid_Chi2']
        })
        
        # Store FJC Gaussian fit results
        fjc_gaussian_fit_results.append({
            'N': N,
            'Model': 'FJC',
            'Mean_FJC': np.mean(fjc_data) if len(fjc_data) > 0 else np.nan,
            'MeanDiff_FJC': fjc_metrics_gaussian['MeanDiff'],
            'Std_FJC': np.std(fjc_data) if len(fjc_data) > 0 else np.nan,
            'StdRatio_FJC': fjc_metrics_gaussian['StdRatio'],
            'KL_FJC': fjc_metrics_gaussian['KL'],
            'KS_FJC': fjc_metrics_gaussian['KS_stat'],
            'Effective_Samples_FJC': fjc_metrics_gaussian['Effective_Samples'],
            'Valid_Chi2_FJC': fjc_metrics_gaussian['Valid_Chi2']
        })
        
        fjc_gaussian_fit_results.append({
            'N': N,
            'Model': 'SAW',
            'Mean_SAW': np.average(saw_data, weights=saw_weights) if len(saw_data) > 0 and saw_weights is not None else np.mean(saw_data) if len(saw_data) > 0 else np.nan,
            'MeanDiff_SAW': saw_metrics_gaussian['MeanDiff'],
            'Std_SAW': np.sqrt(np.average((saw_data - np.average(saw_data, weights=saw_weights))**2, weights=saw_weights)) if len(saw_data) > 0 and saw_weights is not None else np.std(saw_data) if len(saw_data) > 0 else np.nan,
            'StdRatio_SAW': saw_metrics_gaussian['StdRatio'],
            'KL_SAW': saw_metrics_gaussian['KL'],
            'KS_SAW': saw_metrics_gaussian['KS_stat'],
            'Effective_Samples_SAW': saw_metrics_gaussian['Effective_Samples'],
            'Valid_Chi2_SAW': saw_metrics_gaussian['Valid_Chi2']
        })
        
        fjc_gaussian_fit_results.append({
            'N': N,
            'Model': 'WLC',
            'Mean_WLC': np.mean(wlc_data) if len(wlc_data) > 0 else np.nan,
            'MeanDiff_WLC': wlc_metrics_gaussian['MeanDiff'],
            'Std_WLC': np.std(wlc_data) if len(wlc_data) > 0 else np.nan,
            'StdRatio_WLC': wlc_metrics_gaussian['StdRatio'],
            'KL_WLC': wlc_metrics_gaussian['KL'],
            'KS_WLC': wlc_metrics_gaussian['KS_stat'],
            'Effective_Samples_WLC': wlc_metrics_gaussian['Effective_Samples'],
            'Valid_Chi2_WLC': wlc_metrics_gaussian['Valid_Chi2']
        })
        
        # Store Truncated Gaussian fit results
        truncated_gaussian_fit_results.append({
            'N': N,
            'Model': 'FJC',
            'Mean_FJC': np.mean(fjc_data) if len(fjc_data) > 0 else np.nan,
            'MeanDiff_FJC': fjc_metrics_trunc['MeanDiff'],
            'Std_FJC': np.std(fjc_data) if len(fjc_data) > 0 else np.nan,
            'StdRatio_FJC': fjc_metrics_trunc['StdRatio'],
            'KL_FJC': fjc_metrics_trunc['KL'],
            'KS_FJC': fjc_metrics_trunc['KS_stat'],
            'Effective_Samples_FJC': fjc_metrics_trunc['Effective_Samples'],
            'Valid_Chi2_FJC': fjc_metrics_trunc['Valid_Chi2'],
            'Trunc_Mu_FJC': fjc_trunc_mu,
            'Trunc_Sigma_FJC': fjc_trunc_sigma
        })
        
        truncated_gaussian_fit_results.append({
            'N': N,
            'Model': 'SAW',
            'Mean_SAW': np.average(saw_data, weights=saw_weights) if len(saw_data) > 0 and saw_weights is not None else np.mean(saw_data) if len(saw_data) > 0 else np.nan,
            'MeanDiff_SAW': saw_metrics_trunc['MeanDiff'],
            'Std_SAW': np.sqrt(np.average((saw_data - np.average(saw_data, weights=saw_weights))**2, weights=saw_weights)) if len(saw_data) > 0 and saw_weights is not None else np.std(saw_data) if len(saw_data) > 0 else np.nan,
            'StdRatio_SAW': saw_metrics_trunc['StdRatio'],
            'KL_SAW': saw_metrics_trunc['KL'],
            'KS_SAW': saw_metrics_trunc['KS_stat'],
            'Effective_Samples_SAW': saw_metrics_trunc['Effective_Samples'],
            'Valid_Chi2_SAW': saw_metrics_trunc['Valid_Chi2'],
            'Trunc_Mu_SAW': saw_trunc_mu,
            'Trunc_Sigma_SAW': saw_trunc_sigma
        })
        
        truncated_gaussian_fit_results.append({
            'N': N,
            'Model': 'WLC',
            'Mean_WLC': np.mean(wlc_data) if len(wlc_data) > 0 else np.nan,
            'MeanDiff_WLC': wlc_metrics_trunc['MeanDiff'],
            'Std_WLC': np.std(wlc_data) if len(wlc_data) > 0 else np.nan,
            'StdRatio_WLC': wlc_metrics_trunc['StdRatio'],
            'KL_WLC': wlc_metrics_trunc['KL'],
            'KS_WLC': wlc_metrics_trunc['KS_stat'],
            'Effective_Samples_WLC': wlc_metrics_trunc['Effective_Samples'],
            'Valid_Chi2_WLC': wlc_metrics_trunc['Valid_Chi2'],
            'Trunc_Mu_WLC': wlc_trunc_mu,
            'Trunc_Sigma_WLC': wlc_trunc_sigma
        })
        
        # Store comprehensive results with enhanced diagnostics
        results.append({
            'N': N,
            # Enhanced sampling diagnostics
            'FJC_samples': len(fjc_data), 'FJC_attempts': fjc_attempts, 'FJC_acceptance': fjc_acceptance, 'FJC_ESS': fjc_ess,
            'SAW_samples': len(saw_data), 'SAW_attempts': saw_attempts, 'SAW_acceptance': saw_acceptance, 'SAW_ESS': saw_ess_calc,
            'WLC_samples': len(wlc_data), 'WLC_attempts': wlc_attempts, 'WLC_acceptance': wlc_acceptance, 'WLC_ESS': wlc_ess_calc,
            
            # FJC Gaussian parameters
            'FJC_Gaussian_Mean': fjc_mean, 'FJC_Gaussian_Std': fjc_std,
            
            # Truncated Gaussian parameters
            'FJC_Trunc_Mu': fjc_trunc_mu, 'FJC_Trunc_Sigma': fjc_trunc_sigma,
            'SAW_Trunc_Mu': saw_trunc_mu, 'SAW_Trunc_Sigma': saw_trunc_sigma,
            'WLC_Trunc_Mu': wlc_trunc_mu, 'WLC_Trunc_Sigma': wlc_trunc_sigma,
            
            # Error estimates
            'FJC_MeanErr': fjc_mean_err, 'FJC_StdErr': fjc_std_err,
            'SAW_MeanErr': saw_mean_err, 'SAW_StdErr': saw_std_err,
            'WLC_MeanErr': wlc_mean_err, 'WLC_StdErr': wlc_std_err,
            
            # Enhanced fit metrics for Fourier
            'KL_FJC_Fourier': fjc_metrics_fourier['KL'], 'KS_FJC_Fourier': fjc_metrics_fourier['KS_stat'], 'Chi2_FJC_Fourier': fjc_metrics_fourier['Chi2'],
            'Reduced_Chi2_FJC_Fourier': fjc_metrics_fourier['Reduced_Chi2'], 'DOF_FJC_Fourier': fjc_metrics_fourier['DOF'],
            'MeanDiff_FJC_Fourier': fjc_metrics_fourier['MeanDiff'], 'StdRatio_FJC_Fourier': fjc_metrics_fourier['StdRatio'],
            'Valid_Chi2_FJC_Fourier': fjc_metrics_fourier['Valid_Chi2'], 'Effective_Samples_FJC_Fourier': fjc_metrics_fourier['Effective_Samples'],
            
            'KL_SAW_Fourier': saw_metrics_fourier['KL'], 'KS_SAW_Fourier': saw_metrics_fourier['KS_stat'], 'Chi2_SAW_Fourier': saw_metrics_fourier['Chi2'],
            'Reduced_Chi2_SAW_Fourier': saw_metrics_fourier['Reduced_Chi2'], 'DOF_SAW_Fourier': saw_metrics_fourier['DOF'],
            'MeanDiff_SAW_Fourier': saw_metrics_fourier['MeanDiff'], 'StdRatio_SAW_Fourier': saw_metrics_fourier['StdRatio'],
            'Valid_Chi2_SAW_Fourier': saw_metrics_fourier['Valid_Chi2'], 'Effective_Samples_SAW_Fourier': saw_metrics_fourier['Effective_Samples'],
            
            'KL_WLC_Fourier': wlc_metrics_fourier['KL'], 'KS_WLC_Fourier': wlc_metrics_fourier['KS_stat'], 'Chi2_WLC_Fourier': wlc_metrics_fourier['Chi2'],
            'Reduced_Chi2_WLC_Fourier': wlc_metrics_fourier['Reduced_Chi2'], 'DOF_WLC_Fourier': wlc_metrics_fourier['DOF'],
            'MeanDiff_WLC_Fourier': wlc_metrics_fourier['MeanDiff'], 'StdRatio_WLC_Fourier': wlc_metrics_fourier['StdRatio'],
            'Valid_Chi2_WLC_Fourier': wlc_metrics_fourier['Valid_Chi2'], 'Effective_Samples_WLC_Fourier': wlc_metrics_fourier['Effective_Samples']
        })
        
        # ENHANCED PLOT FORMATTING WITH LARGER FONTS AND BETTER VISIBILITY
        quality_indicator = ""
        if saw_metrics_fourier['Valid_Chi2'] and saw_metrics_fourier['KL'] < 0.1:
            quality_indicator = "✓"
        elif not saw_metrics_fourier['Valid_Chi2']:
            quality_indicator = "⚠️"
        else:
            quality_indicator = "∼"
            
        # Set title with larger font
        ax.set_title(f'N = {N} {quality_indicator}\nFJC:{len(fjc_data):d} SAW:{len(saw_data):d} WLC:{len(wlc_data):d}',
                    fontsize=20, fontweight='bold', pad=20)
        
        # Set axis labels with larger bold font
        ax.set_xlabel('x (μm)', fontsize=18, fontweight='bold', labelpad=10)
        ax.set_ylabel('P(x)', fontsize=18, fontweight='bold', labelpad=10)
        
        # Set axis limits and grid
        ax.set_xlim(0, L)
        ax.set_ylim(0, 2.75)
        ax.grid(True, alpha=0.3)
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Legend with larger font and better positioning
        if idx == 0:
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9, bbox_to_anchor=(1, 1))
    
    # Remove empty subplots
    for j in range(len(N_values), len(axes)):
        fig.delaxes(axes[j])
        
    fig.tight_layout()
    fig.savefig('Enhanced_Combined_FJC_SAW_WLC_Analysis_with_All_Fits.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create and save enhanced results table
    df = pd.DataFrame(results)
    
    # Create and save Fourier fit metrics
    fourier_df = pd.DataFrame(fourier_fit_results)
    fourier_df.to_csv('fourier_fit_metrics.csv', index=False)
    print(f"\nFourier fit metrics saved to 'fourier_fit_metrics.csv'")
    
    # Create and save FJC Gaussian fit metrics
    fjc_gaussian_df = pd.DataFrame(fjc_gaussian_fit_results)
    fjc_gaussian_df.to_csv('fjc_gaussian_fit_metrics.csv', index=False)
    print(f"FJC Gaussian fit metrics saved to 'fjc_gaussian_fit_metrics.csv'")
    
    # Create and save Truncated Gaussian fit metrics
    truncated_gaussian_df = pd.DataFrame(truncated_gaussian_fit_results)
    truncated_gaussian_df.to_csv('truncated_gaussian_fit_metrics.csv', index=False)
    print(f"Truncated Gaussian fit metrics saved to 'truncated_gaussian_fit_metrics.csv'")
    
    # Create metrics comparison plots
    print("\nCreating fit metrics comparison plots...")
    plot_fit_metrics_comparison(fourier_df, fjc_gaussian_df, truncated_gaussian_df)
    
    print("\n" + "="*120)
    print("ENHANCED COMBINED FJC, SAW, and WLC ANALYSIS RESULTS")
    print("="*120)
    
    # Enhanced sampling diagnostics
    print("\nENHANCED SAMPLING SUMMARY:")
    print("-" * 130)
    print("N  Model  Samples/Attempts  Acceptance   ESS      MeanErr    StdErr    EffSamples")
    print("-" * 130)
    for _, row in df.iterrows():
        N = row['N']
        print(f"{N:2d} FJC   {row['FJC_samples']:6d}/{row['FJC_attempts']:8d}   {row['FJC_acceptance']:10.4f}   {row['FJC_ESS']:8.0f}   {row['FJC_MeanErr']:9.6f}   {row['FJC_StdErr']:9.6f}   {row['Effective_Samples_FJC_Fourier']:8.0f}")
        print(f"    SAW   {row['SAW_samples']:6d}/{row['SAW_attempts']:8d}   {row['SAW_acceptance']:10.4f}   {row['SAW_ESS']:8.1f}   {row['SAW_MeanErr']:9.6f}   {row['SAW_StdErr']:9.6f}   {row['Effective_Samples_SAW_Fourier']:8.1f}")
        print(f"    WLC   {row['WLC_samples']:6d}/{row['WLC_attempts']:8d}   {row['WLC_acceptance']:10.4f}   {row['WLC_ESS']:8.0f}   {row['WLC_MeanErr']:9.6f}   {row['WLC_StdErr']:9.6f}   {row['Effective_Samples_WLC_Fourier']:8.0f}")
        print()
        
    # Enhanced fit metrics summary
    print("\nENHANCED FIT METRICS SUMMARY (Fourier):")
    print("-" * 130)
    print("N  Model  KL-divergence  KS-statistic  Reduced-χ²  Mean-Diff  Std-Ratio  Valid-χ²  EffSamples")
    print("-" * 130)
    for _, row in df.iterrows():
        N = row['N']
        valid_fjc = "✓" if row['Valid_Chi2_FJC_Fourier'] else "✗"
        valid_saw = "✓" if row['Valid_Chi2_SAW_Fourier'] else "✗"
        valid_wlc = "✓" if row['Valid_Chi2_WLC_Fourier'] else "✗"
        
        print(f"{N:2d} FJC   {row['KL_FJC_Fourier']:12.6f}   {row['KS_FJC_Fourier']:12.6f}   {row['Reduced_Chi2_FJC_Fourier']:10.4f}   {row['MeanDiff_FJC_Fourier']:9.6f}   {row['StdRatio_FJC_Fourier']:9.4f}    {valid_fjc}     {row['Effective_Samples_FJC_Fourier']:8.0f}")
        print(f"    SAW   {row['KL_SAW_Fourier']:12.6f}   {row['KS_SAW_Fourier']:12.6f}   {row['Reduced_Chi2_SAW_Fourier']:10.4f}   {row['MeanDiff_SAW_Fourier']:9.6f}   {row['StdRatio_SAW_Fourier']:9.4f}    {valid_saw}     {row['Effective_Samples_SAW_Fourier']:8.1f}")
        print(f"    WLC   {row['KL_WLC_Fourier']:12.6f}   {row['KS_WLC_Fourier']:12.6f}   {row['Reduced_Chi2_WLC_Fourier']:10.4f}   {row['MeanDiff_WLC_Fourier']:9.6f}   {row['StdRatio_WLC_Fourier']:9.4f}    {valid_wlc}     {row['Effective_Samples_WLC_Fourier']:8.0f}")
        print()
        
    # Save enhanced results
    df.to_csv('enhanced_combined_fjc_saw_wlc_results_with_all_fits.csv', index=False)
    print(f"\nEnhanced results saved to 'enhanced_combined_fjc_saw_wlc_results_with_all_fits.csv'")
    
    return df, fourier_df, fjc_gaussian_df, truncated_gaussian_df

# ===========================================
# RUN ENHANCED ANALYSIS
# ===========================================
if __name__ == "__main__":
    print("Running ENHANCED comprehensive FJC, SAW, and WLC analysis...")
    print("IMPROVED ADAPTIVE SAW SAMPLING STRATEGY:")
    print("  N < 30: Improved Simple MC (clearance-optimized steps)")
    print("  30 ≤ N < 40: Improved PERM (progressive parameter tightening)")
    print("  N ≥ 40: Improved Pivot (adaptive angles + guaranteed samples)")
    print("  ADAPTIVE SAMPLE SIZING: 2000 (N<30), 2500 (N=30-39), 3000 (N=40-49), 3500 (N≥50)")
    print("WLC: Adaptive persistence length + enhanced sampling")
    print("Fit Metrics: Higher robustness thresholds + confidence intervals")
    print("Visualization: Quality indicators + enhanced error bands")
    print("BOUNDARY ENFORCEMENT: P(x) = 0 at x=0 and x=L")
    print("PERFORMANCE OPTIMIZATION: Adaptive resource allocation")
    print("NEW: Improved algorithms with better convergence properties")
    print("NEW: Three separate CSV files for different fitting approaches")
    print("NEW: Comprehensive metrics comparison plots with all fitting methods")
    print("=" * 80)
    
    # Run with enhanced parameters
    enhanced_results, fourier_metrics, fjc_gaussian_metrics, truncated_gaussian_metrics = enhanced_analyze_combined_models(
        N_values, a=a, lp=lp, x0=x0, L=L,
        n_walkers_fjc=100000,  # Reduced from 200000
        n_walkers_saw=2000,    # Base value - adaptive sizing applied internally
        n_walkers_wlc=30000    # Reduced from 50000
    )
    
    print("\nEnhanced analysis complete!")
    print("Key improvements implemented:")
    print("✅ IMPROVED SAW SAMPLING: Adaptive sample sizing based on chain length")
    print("✅ Simple MC: Clearance-optimized step selection")
    print("✅ PERM: Progressive parameter tightening and weighted step selection") 
    print("✅ Pivot: Adaptive angle adjustment and guaranteed sample collection")
    print("✅ Boundary enforcement: P(x) = 0 at x=0 and x=L")
    print("✅ Separate CSV files: Fourier, FJC Gaussian, and Truncated Gaussian fit metrics")
    print("✅ Fit metrics plots: 6 key metrics comparison across all fitting methods")
    print("✅ N=30 transition: Special PERM tuning with adaptive parameters")
    print("✅ WLC improvements: Adaptive persistence length + progress tracking")
    print("✅ Robust statistics: Higher validation thresholds + effective sample tracking")
    print("✅ Quality indicators: Visual feedback on plot quality")
    print("✅ Comprehensive diagnostics: Enhanced reporting at all stages")
    print("✅ All three fitting curves displayed on each subplot")
    print("✅ IMPROVED VISUALIZATION: Larger fonts (18pt bold axis labels)")
    print("✅ IMPROVED VISUALIZATION: Thicker lines for better visibility")
    print("✅ IMPROVED VISUALIZATION: SAW bootstrap error bands clearly visible")
    print("✅ IMPROVED VISUALIZATION: Better legend positioning and sizing")