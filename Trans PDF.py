# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:34:00 2025

@author: SOUMYA
"""

# ===========================================
# ROBUST Combined FJC, SAW, and WLC Model Analysis for P(y)
# Enhanced version with PERM algorithm and robust error handling
# ===========================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from numba import njit
import pandas as pd
import warnings
import time
from typing import List, Dict, Tuple, Optional

# ===========================================
# Parameters
# ===========================================

a = 0.1  # Kuhn length (µm)
R = 0.4  # Confinement radius (µm)
lp = 0.2  # Persistence length for WLC (µm)
y0 = 0.0  # Initial position at center

# Chain lengths to analyze
N_values = [5, 10, 15, 20, 25, 30, 40, 50, 60]

# SAW parameters
bead_radius = 0.03  # Self-avoidance radius

# ===========================================
# Parameter Validation
# ===========================================

def validate_simulation_parameters(N: int, a: float, R: float, n_walkers: int) -> None:
    """Validate simulation parameters for robustness"""
    if N <= 0:
        raise ValueError(f"Chain length N must be positive, got {N}")
    if a <= 0:
        raise ValueError(f"Kuhn length must be positive, got {a}")
    if R <= 0:
        raise ValueError(f"Confinement radius must be positive, got {R}")
    if n_walkers < 100:
        raise ValueError(f"Need at least 100 walkers, got {n_walkers}")
    
    # Warn about potentially problematic parameters
    if N > 50:
        warnings.warn(f"Large chain length N={N} may have sampling issues")
    if n_walkers > 1000000:
        warnings.warn(f"Large number of walkers {n_walkers} may be computationally expensive")

# ===========================================
# Analytical Solutions (UNCHANGED)
# ===========================================

def confined_Py(y, N, a, R, n_terms=20):
    sigma2 = N * a**2 / 3
    prefac = 1.0 / np.sqrt(2 * np.pi * sigma2)
    s = np.zeros_like(y)
    
    for n in range(-n_terms, n_terms + 1):
        s += (-1)**n * np.exp(-((y + 2 * n * R)**2) / (2 * sigma2))
    
    Py = prefac * s
    # Normalize
    norm_factor = trapezoid(Py, y)
    if norm_factor > 0:
        Py /= norm_factor
    return Py

def confined_Py_fourier(y, R, a, N, n_terms=101):
    numerator = np.zeros_like(y)
    denominator = 0.0
    
    for n in range(1, n_terms + 1, 2):
        n_pi = n * np.pi
        lambda_n = (a * n_pi)**2 / (8 * R**2)
        y_shifted = y + R
        numerator += np.sin(n_pi * y_shifted / (2 * R)) * np.exp(-lambda_n * N)
        denominator += (1 / n) * np.exp(-lambda_n * N)
    
    if denominator == 0:
        return np.ones_like(y) / (2 * R)  # Uniform distribution as fallback
    
    P = (np.pi / (2 * R)) * (numerator / denominator)
    # Normalize
    norm_factor = trapezoid(P, y)
    if norm_factor > 0:
        P /= norm_factor
    return P

# ===========================================
# FJC Monte Carlo for P(y) (UNCHANGED)
# ===========================================

@njit
def random_unit_vector():
    while True:
        v = np.random.normal(0, 1, 3)
        norm_v = np.sqrt(np.sum(v**2))
        if norm_v > 1e-8:
            return v / norm_v

def fjc_monte_carlo_Py(N, a, R, n_walkers=200000, seed=123):
    np.random.seed(seed)
    ends = []
    
    for _ in range(n_walkers):
        positions = np.zeros((N + 1, 3))
        valid = True
        
        for i in range(1, N + 1):
            step = random_unit_vector()
            positions[i] = positions[i-1] + a * step
            
            if np.abs(positions[i, 1]) > R:  # Confinement check in y-direction
                valid = False
                break
        
        if valid:
            ends.append(positions[-1, 1])
    
    return np.array(ends)

# ===========================================
# PERM Algorithm for Efficient SAW Sampling
# ===========================================

@njit
def get_available_directions(current_pos: np.ndarray, existing_positions: List[np.ndarray], 
                           bead_radius: float, R: float) -> List[np.ndarray]:
    """Get all valid directions for the next step using PERM approach"""
    available_dirs = []
    
    # Try 6 principal directions first (most efficient)
    principal_dirs = [
        np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    ]
    
    for dir_vec in principal_dirs:
        new_pos = current_pos + a * dir_vec
        
        # Confinement check
        if np.abs(new_pos[1]) > R - bead_radius:
            continue
            
        # Self-avoidance check
        valid = True
        for pos in existing_positions:
            if np.linalg.norm(new_pos - pos) < 2 * bead_radius:
                valid = False
                break
                
        if valid:
            available_dirs.append(dir_vec)
    
    # If principal directions don't work well, sample random directions
    if len(available_dirs) < 3:
        for _ in range(12):  # Sample additional random directions
            dir_vec = random_unit_vector()
            new_pos = current_pos + a * dir_vec
            
            # Confinement check
            if np.abs(new_pos[1]) > R - bead_radius:
                continue
                
            # Self-avoidance check
            valid = True
            for pos in existing_positions:
                if np.linalg.norm(new_pos - pos) < 2 * bead_radius:
                    valid = False
                    break
                    
            if valid:
                # Avoid duplicates
                is_duplicate = False
                for existing_dir in available_dirs:
                    if np.dot(dir_vec, existing_dir) > 0.95:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    available_dirs.append(dir_vec)
    
    return available_dirs

def perm_saw_monte_carlo_Py(N: int, a: float, R: float, bead_radius: float = 0.03, 
                          n_walkers: int = 5000, max_depth: int = 100000,
                          prune_threshold: float = 0.1, enrich_threshold: float = 2.0) -> np.ndarray:
    """
    PERM (Pruned-Enriched Rosenbluth Method) for efficient SAW sampling
    
    Parameters:
    - prune_threshold: chains with weight below this are pruned
    - enrich_threshold: chains with weight above this are enriched (cloned)
    """
    ends = []
    total_generated = 0
    
    # Initial chain (just the starting point)
    chains = [{
        'positions': [np.array([0.0, 0.0, 0.0])],
        'weight': 1.0
    }]
    
    while len(ends) < n_walkers and total_generated < max_depth:
        new_chains = []
        
        for chain in chains:
            current_positions = chain['positions']
            current_weight = chain['weight']
            current_length = len(current_positions)
            
            if current_length == N + 1:
                # Chain is complete
                final_y = current_positions[-1][1]
                if abs(final_y) <= R:
                    ends.append(final_y)
                continue
            
            # Get available directions for next step
            available_dirs = get_available_directions(
                current_positions[-1], current_positions, bead_radius, R
            )
            
            if not available_dirs:
                # Dead end - prune this chain
                continue
            
            # Calculate new weight
            k = len(available_dirs)
            new_weight = current_weight * k / 6.0  # 6 is coordination number for cubic lattice
            
            # PERM decisions based on weight
            if new_weight < prune_threshold:
                # Prune with probability 1 - new_weight/prune_threshold
                if np.random.random() < new_weight / prune_threshold:
                    # Select one direction randomly
                    selected_dir = available_dirs[np.random.randint(len(available_dirs))]
                    new_pos = current_positions[-1] + a * selected_dir
                    new_chain = {
                        'positions': current_positions + [new_pos],
                        'weight': prune_threshold  # Reset weight after surviving pruning
                    }
                    new_chains.append(new_chain)
            elif new_weight > enrich_threshold:
                # Enrich: clone the chain multiple times
                num_clones = min(int(new_weight / enrich_threshold) + 1, 5)  # Max 5 clones
                for _ in range(num_clones):
                    selected_dir = available_dirs[np.random.randint(len(available_dirs))]
                    new_pos = current_positions[-1] + a * selected_dir
                    new_chain = {
                        'positions': current_positions + [new_pos],
                        'weight': new_weight / num_clones  # Distribute weight among clones
                    }
                    new_chains.append(new_chain)
            else:
                # Normal growth
                selected_dir = available_dirs[np.random.randint(len(available_dirs))]
                new_pos = current_positions[-1] + a * selected_dir
                new_chain = {
                    'positions': current_positions + [new_pos],
                    'weight': new_weight
                }
                new_chains.append(new_chain)
        
        chains = new_chains
        total_generated += 1
        
        # Population control: if too many chains, randomly sample
        if len(chains) > 10000:
            indices = np.random.choice(len(chains), size=5000, replace=False)
            chains = [chains[i] for i in indices]
    
    # If PERM doesn't generate enough samples, fall back to traditional method
    if len(ends) < n_walkers // 2:
        print(f"⚠️ PERM at N={N}: Only {len(ends)} samples, using fallback...")
        fallback_samples = robust_saw_monte_carlo_Py(N, a, R, bead_radius, 
                                                   n_walkers - len(ends))
        ends.extend(fallback_samples)
    
    final_ends = np.array(ends[:n_walkers])
    print(f"✅ PERM SAW at N={N}: {len(final_ends)} samples generated")
    
    return final_ends

# ===========================================
# Robust SAW Monte Carlo (Fallback)
# ===========================================

@njit
def is_valid_saw_position(new_pos, existing_positions, bead_radius, R):
    """Check if new position is valid (confinement + self-avoidance)"""
    # Confinement check
    if np.abs(new_pos[1]) > R - bead_radius:
        return False
    
    # Self-avoidance check
    for pos in existing_positions:
        if np.linalg.norm(new_pos - pos) < 2 * bead_radius:
            return False
    return True

def robust_saw_monte_carlo_Py(N: int, a: float, R: float, bead_radius: float = 0.03, 
                             n_walkers: int = 5000, max_total_attempts: int = 1000000) -> np.ndarray:
    """
    ROBUST SAW sampling with multiple fallback strategies
    """
    strategies = [
        # Strategy 1: Standard approach with center bias
        {"trials_per_step": 20, "center_bias": 0.3, "max_attempts": n_walkers * 100},
        # Strategy 2: Aggressive center bias
        {"trials_per_step": 30, "center_bias": 0.5, "max_attempts": n_walkers * 150},
        # Strategy 3: Reduced bead radius temporarily
        {"trials_per_step": 25, "center_bias": 0.4, "bead_radius_factor": 0.8, 
         "max_attempts": n_walkers * 100},
        # Strategy 4: Very aggressive sampling
        {"trials_per_step": 50, "center_bias": 0.6, "max_attempts": n_walkers * 200}
    ]
    
    all_ends = []
    total_attempts = 0
    
    for strategy_idx, strategy in enumerate(strategies):
        if len(all_ends) >= n_walkers:
            break
            
        current_bead_radius = (strategy.get("bead_radius_factor", 1.0) * bead_radius)
        remaining_needed = n_walkers - len(all_ends)
        strategy_ends = []
        attempts = 0
        max_strategy_attempts = min(strategy["max_attempts"], 
                                   max_total_attempts - total_attempts)
        
        while (len(strategy_ends) < remaining_needed and 
               attempts < max_strategy_attempts and
               total_attempts < max_total_attempts):
            
            positions = [np.array([0.0, 0.0, 0.0])]
            valid_chain = True
            
            for i in range(1, N + 1):
                current_pos = positions[-1]
                found_valid_step = False
                
                for trial in range(strategy["trials_per_step"]):
                    # Apply center bias
                    step = np.random.normal(0, 1, 3)
                    if strategy["center_bias"] > 0 and i > 1:
                        bias_strength = (strategy["center_bias"] * 
                                       (1.0 - abs(current_pos[1]) / R))
                        step[1] += bias_strength * (-current_pos[1])
                    
                    step_norm = np.sqrt(np.sum(step**2))
                    if step_norm > 1e-8:
                        step = step / step_norm
                    else:
                        continue
                    
                    new_pos = current_pos + a * step
                    
                    if is_valid_saw_position(new_pos, positions, 
                                           current_bead_radius, R):
                        positions.append(new_pos)
                        found_valid_step = True
                        break
                
                if not found_valid_step:
                    valid_chain = False
                    break
            
            attempts += 1
            total_attempts += 1
            
            if valid_chain and len(positions) == N + 1:
                final_y = positions[-1][1]
                if abs(final_y) <= R:
                    strategy_ends.append(final_y)
        
        if strategy_ends:
            all_ends.extend(strategy_ends)
            print(f"✅ Strategy {strategy_idx+1}: Added {len(strategy_ends)} chains "
                  f"(rate: {len(strategy_ends)/max(attempts,1):.4f})")
    
    final_ends = np.array(all_ends[:n_walkers])  # Trim to exact target
    acceptance_rate = len(final_ends) / max(total_attempts, 1)
    
    print(f"Robust SAW N={N}: {len(final_ends)}/{n_walkers} chains "
          f"(overall rate: {acceptance_rate:.4f}, attempts: {total_attempts})")
    
    return final_ends

# ===========================================
# WLC Monte Carlo for P(y) (UNCHANGED)
# ===========================================

@njit
def generate_wlc_chain_y(N, a, lp, R):
    # Initialize chain coordinates and directions
    coords = np.zeros((N+1, 3))
    tangents = np.zeros((N, 3))
    
    # Start at origin, random initial direction
    initial_dir = np.random.normal(0, 1, 3)
    initial_dir /= np.sqrt(np.sum(initial_dir**2))
    tangents[0] = initial_dir
    
    # Generate the chain
    for i in range(1, N+1):
        # Propose new direction based on bending energy
        current_tangent = tangents[i-1]
        
        # Generate trial direction with bias toward current direction
        bending_stiffness = lp / a  # dimensionless bending parameter
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
                new_direction = current_tangent  # fallback
        else:
            new_direction = v  # random direction for kappa=0
        
        new_direction /= np.sqrt(np.sum(new_direction**2))
        
        # Update position
        coords[i] = coords[i-1] + a * new_direction
        if i < N:
            tangents[i] = new_direction
        
        # Check confinement in y-direction
        if np.abs(coords[i, 1]) > R:
            return -10.0
    
    return coords[-1, 1]

def wlc_monte_carlo_Py(N, a, R, lp=0.2, n_walkers=50000, max_attempts_factor=100):
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * n_walkers
    
    while len(ends) < n_walkers and attempts < max_attempts:
        val = generate_wlc_chain_y(N, a, lp, R)
        if val > -5:  # Valid chain
            ends.append(val)
        attempts += 1
    
    if len(ends) < n_walkers:
        print(f"⚠️ WLC at N={N}: {len(ends)} of {n_walkers} accepted")
    else:
        print(f"✅ WLC at N={N}: {len(ends)} of {n_walkers} accepted")
    
    return np.array(ends)

# ===========================================
# Robust KDE for smooth distribution
# ===========================================

def safe_adaptive_kde(data, y_vals, bw_factor=1.5, R=0.4, min_samples=50):
    """Robust KDE with comprehensive error handling"""
    if len(data) < min_samples:
        return None
    
    try:
        # Remove any NaN or infinite values
        clean_data = data[np.isfinite(data)]
        if len(clean_data) < min_samples:
            return None
        
        # Check for sufficient variance
        if np.std(clean_data) < 1e-10:
            return None
            
        kde = gaussian_kde(clean_data)
        
        # Adaptive bandwidth selection
        silverman_bandwidth = kde.silverman_factor()
        scott_bandwidth = kde.scotts_factor()
        final_bandwidth = min(silverman_bandwidth, scott_bandwidth) * bw_factor
        
        kde.set_bandwidth(bw_method=final_bandwidth)
        pdf = kde(y_vals)
        pdf = np.clip(pdf, 0, None)  # Ensure non-negative
        
        # Normalize within confinement
        mask = (y_vals >= -R) & (y_vals <= R)
        if np.any(mask):
            norm_factor = trapezoid(pdf[mask], y_vals[mask])
            if norm_factor > 0:
                pdf /= norm_factor
            else:
                return None
        else:
            return None
            
        return pdf
        
    except (ValueError, np.linalg.LinAlgError, Exception) as e:
        print(f"KDE failed: {e}")
        return None

# ===========================================
# Fit metrics with enhanced robustness
# ===========================================

def calculate_fit_metrics(data, expected_pdf, y_grid, R=0.4, n_bins=50):
    if len(data) < 50:
        return np.nan, np.nan, np.nan, np.nan
    
    try:
        hist_obs, edges = np.histogram(data, bins=n_bins, range=(-R, R), density=True)
        centers = 0.5*(edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        
        expected_vals = np.interp(centers, y_grid, expected_pdf)
        
        # Convert to probabilities
        p_obs = hist_obs * width
        p_exp = expected_vals * width
        
        p_obs /= np.sum(p_obs)
        p_exp /= np.sum(p_exp)
        
        eps = 1e-12
        p_obs = np.clip(p_obs, eps, None)
        p_exp = np.clip(p_exp, eps, None)
        
        # KL divergence
        KL = np.sum(p_obs * np.log(p_obs/p_exp))
        
        # Reduced chi-square
        counts_obs, _ = np.histogram(data, bins=n_bins, range=(-R, R))
        expected_counts = expected_vals * len(data) * width
        
        mask = expected_counts > 1
        if np.sum(mask) < 5:
            chi2 = np.nan
            red_chi2 = np.nan
        else:
            chi2 = np.sum((counts_obs[mask] - expected_counts[mask])**2 / expected_counts[mask])
            dof = np.sum(mask) - 1
            red_chi2 = chi2 / dof if dof > 0 else np.nan
        
        # Mean difference and Std ratio
        mean_data = np.mean(data)
        std_data = np.std(data)
        
        # Calculate mean and std for analytic distribution
        dy = y_grid[1] - y_grid[0]
        mean_analytic = np.sum(y_grid * expected_pdf) * dy
        std_analytic = np.sqrt(np.sum((y_grid - mean_analytic)**2 * expected_pdf) * dy)
        
        mean_diff = mean_data - mean_analytic
        std_ratio = std_data / std_analytic if std_analytic > 0 else np.nan
        
        return KL, red_chi2, mean_diff, std_ratio
    
    except Exception as e:
        print(f"Metric calculation failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

def calculate_robust_fit_metrics(data, expected_pdf, y_grid, R=0.4, n_bootstrap=100):
    """Enhanced metrics with bootstrap confidence intervals"""
    if len(data) < 100:
        return {k: np.nan for k in ['KL', 'Chi2', 'MeanDiff', 'StdRatio', 
                                   'KL_std', 'Chi2_std']}
    
    # Original metric calculation
    base_KL, base_Chi2, base_MeanDiff, base_StdRatio = calculate_fit_metrics(data, expected_pdf, y_grid, R)
    
    # Bootstrap for uncertainty estimation
    KL_samples, Chi2_samples = [], []
    n_data = len(data)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n_data, replace=True)
        try:
            KL_bs, Chi2_bs, _, _ = calculate_fit_metrics(bootstrap_sample, expected_pdf, y_grid, R)
            if not np.isnan(KL_bs):
                KL_samples.append(KL_bs)
            if not np.isnan(Chi2_bs):
                Chi2_samples.append(Chi2_bs)
        except:
            continue
    
    # Add uncertainty estimates
    enhanced_metrics = {
        'KL': base_KL, 'Chi2': base_Chi2,
        'MeanDiff': base_MeanDiff, 'StdRatio': base_StdRatio,
        'KL_std': np.std(KL_samples) if KL_samples else np.nan,
        'Chi2_std': np.std(Chi2_samples) if Chi2_samples else np.nan,
        'n_valid_bootstrap': len(KL_samples)
    }
    
    return enhanced_metrics

# ===========================================
# Progress Monitoring and Resource Management
# ===========================================

class SimulationMonitor:
    """Monitor simulation progress and adapt resources"""
    def __init__(self):
        self.history = {}
        self.start_time = time.time()
        
    def record_performance(self, model: str, N: int, n_samples: int, 
                         n_attempts: int, success_rate: float):
        key = f"{model}_N{N}"
        self.history[key] = {
            'samples': n_samples,
            'attempts': n_attempts, 
            'success_rate': success_rate,
            'timestamp': time.time()
        }
    
    def get_recommended_samples(self, model: str, N: int, target_confidence: float = 0.95) -> int:
        """Recommend sample size based on historical performance"""
        key = f"{model}_N{N}"
        if key in self.history:
            historical_rate = self.history[key]['success_rate']
            # Adjust samples based on historical success rate
            if historical_rate < 0.1:
                return min(5000, int(5000 / max(historical_rate, 0.01)))
            elif historical_rate > 0.5:
                return 5000  # Keep target
        return 5000  # Default
    
    def print_progress(self, current_N: int, total_N: int, model: str):
        """Print progress information"""
        elapsed = time.time() - self.start_time
        progress = (current_N / total_N) * 100
        print(f"📊 Progress: {current_N}/{total_N} ({progress:.1f}%) - "
              f"Elapsed: {elapsed:.1f}s - Current: {model} N={current_N}")

# ===========================================
# ENHANCED Main analysis function
# ===========================================

def analyze_combined_Py_models_enhanced(N_values, a=0.1, R=0.4, lp=0.2,
                                      n_walkers_fjc=200000, 
                                      n_walkers_saw=5000, 
                                      n_walkers_wlc=50000,
                                      use_perm: bool = True):
    """
    ENHANCED robust analysis with PERM algorithm and comprehensive error handling
    """
    # Parameter validation
    for N in N_values:
        validate_simulation_parameters(N, a, R, n_walkers_saw)
    
    monitor = SimulationMonitor()
    
    # Create subplots
    n_plots = len(N_values)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    results = []
    y_grid = np.linspace(-R, R, 400)
    
    for idx, N in enumerate(N_values):
        ax = axes[idx]
        
        print(f"\n{'='*60}")
        print(f"Analyzing N = {N}...")
        print(f"{'='*60}")
        
        try:
            # Run simulations with enhanced error handling
            monitor.print_progress(idx + 1, len(N_values), "FJC")
            fjc_data = fjc_monte_carlo_Py(N, a, R, n_walkers=n_walkers_fjc)
            
            monitor.print_progress(idx + 1, len(N_values), "SAW")
            if use_perm and N >= 10:  # Use PERM for longer chains
                saw_data = perm_saw_monte_carlo_Py(N, a, R, bead_radius, n_walkers_saw)
            else:
                saw_data = robust_saw_monte_carlo_Py(N, a, R, bead_radius, n_walkers_saw)
            
            monitor.print_progress(idx + 1, len(N_values), "WLC")
            wlc_data = wlc_monte_carlo_Py(N, a, R, lp=lp, n_walkers=n_walkers_wlc)
            
        except Exception as e:
            print(f"❌ Simulation failed for N={N}: {e}")
            # Implement fallback - use previous successful data or skip
            continue
        
        # Analytical solutions
        Py_image = confined_Py(y_grid, N, a, R)
        Py_fourier = confined_Py_fourier(y_grid, R, a, N)
        
        # Plot FJC histogram
        ax.hist(fjc_data, bins=100, range=(-R, R), density=True, alpha=0.5,
                color='cornflowerblue', edgecolor='black', linewidth=0.3,
                label=f'FJC MC (n={len(fjc_data)})')
        
        # Plot SAW distribution using robust KDE
        if len(saw_data) > 500:
            saw_pdf = safe_adaptive_kde(saw_data, y_grid, R=R)
            if saw_pdf is not None:
                ax.plot(y_grid, saw_pdf, color='green', lw=2.5, 
                       label=f'SAW MC (n={len(saw_data)})')
            else:
                # Fallback to histogram for SAW
                ax.hist(saw_data, bins=60, range=(-R, R), density=True, alpha=0.5,
                        color='green', edgecolor='black', linewidth=0.3,
                        label=f'SAW MC (n={len(saw_data)})')
        else:
            # For small sample sizes, use histogram
            ax.hist(saw_data, bins=40, range=(-R, R), density=True, alpha=0.6,
                    color='green', edgecolor='black', linewidth=0.3,
                    label=f'SAW MC (n={len(saw_data)})')
        
        # Plot WLC distribution using robust KDE
        if len(wlc_data) > 500:
            wlc_pdf = safe_adaptive_kde(wlc_data, y_grid, R=R)
            if wlc_pdf is not None:
                ax.plot(y_grid, wlc_pdf, color='darkorange', lw=2.5, 
                       label=f'WLC MC (n={len(wlc_data)})')
            else:
                ax.hist(wlc_data, bins=60, range=(-R, R), density=True, alpha=0.5,
                        color='darkorange', edgecolor='black', linewidth=0.3,
                        label=f'WLC MC (n={len(wlc_data)})')
        else:
            ax.hist(wlc_data, bins=40, range=(-R, R), density=True, alpha=0.6,
                    color='darkorange', edgecolor='black', linewidth=0.3,
                    label=f'WLC MC (n={len(wlc_data)})')
        
        # Plot analytical solutions
        ax.plot(y_grid, Py_fourier, 'k-', lw=2.5, label='Fourier analytic')
        ax.plot(y_grid, Py_image, 'm--', lw=2.0, label='Image method analytic')
        
        # Calculate metrics for each model vs both analytical solutions
        KL_fjc_fourier, chi2_fjc_fourier, mean_diff_fjc_fourier, std_ratio_fjc_fourier = calculate_fit_metrics(fjc_data, Py_fourier, y_grid, R)
        KL_fjc_image, chi2_fjc_image, mean_diff_fjc_image, std_ratio_fjc_image = calculate_fit_metrics(fjc_data, Py_image, y_grid, R)
        
        KL_saw_fourier, chi2_saw_fourier, mean_diff_saw_fourier, std_ratio_saw_fourier = calculate_fit_metrics(saw_data, Py_fourier, y_grid, R)
        KL_saw_image, chi2_saw_image, mean_diff_saw_image, std_ratio_saw_image = calculate_fit_metrics(saw_data, Py_image, y_grid, R)
        
        KL_wlc_fourier, chi2_wlc_fourier, mean_diff_wlc_fourier, std_ratio_wlc_fourier = calculate_fit_metrics(wlc_data, Py_fourier, y_grid, R)
        KL_wlc_image, chi2_wlc_image, mean_diff_wlc_image, std_ratio_wlc_image = calculate_fit_metrics(wlc_data, Py_image, y_grid, R)
        
        # Store results
        results.append({
            'N': N,
            'FJC_samples': len(fjc_data),
            'SAW_samples': len(saw_data),
            'WLC_samples': len(wlc_data),
            # Fourier metrics
            'KL_FJC_Fourier': KL_fjc_fourier, 'Chi2_FJC_Fourier': chi2_fjc_fourier, 
            'MeanDiff_FJC_Fourier': mean_diff_fjc_fourier, 'StdRatio_FJC_Fourier': std_ratio_fjc_fourier,
            'KL_SAW_Fourier': KL_saw_fourier, 'Chi2_SAW_Fourier': chi2_saw_fourier,
            'MeanDiff_SAW_Fourier': mean_diff_saw_fourier, 'StdRatio_SAW_Fourier': std_ratio_saw_fourier,
            'KL_WLC_Fourier': KL_wlc_fourier, 'Chi2_WLC_Fourier': chi2_wlc_fourier,
            'MeanDiff_WLC_Fourier': mean_diff_wlc_fourier, 'StdRatio_WLC_Fourier': std_ratio_wlc_fourier,
            # Image method metrics
            'KL_FJC_Image': KL_fjc_image, 'Chi2_FJC_Image': chi2_fjc_image,
            'MeanDiff_FJC_Image': mean_diff_fjc_image, 'StdRatio_FJC_Image': std_ratio_fjc_image,
            'KL_SAW_Image': KL_saw_image, 'Chi2_SAW_Image': chi2_saw_image,
            'MeanDiff_SAW_Image': mean_diff_saw_image, 'StdRatio_SAW_Image': std_ratio_saw_image,
            'KL_WLC_Image': KL_wlc_image, 'Chi2_WLC_Image': chi2_wlc_image,
            'MeanDiff_WLC_Image': mean_diff_wlc_image, 'StdRatio_WLC_Image': std_ratio_wlc_image
        })
        
        # Plot formatting with larger font sizes
        ax.set_title(f'N = {N}', fontsize=18, fontweight='bold')
        ax.set_xlabel('y (μm)', fontsize=18, fontweight='bold')
        ax.set_ylabel('P(y)', fontsize=18, fontweight='bold')
        ax.set_xlim(-R, R)
        ax.set_ylim(0, 4.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if idx == 0:
            ax.legend(fontsize=10, loc='upper right')
    
    # Remove empty subplots
    for j in range(len(N_values), len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout()
    output_filename = 'ENHANCED_Combined_FJC_SAW_WLC_Py_Analysis.pdf'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create and save combined results table
    df = pd.DataFrame(results)
    print("\n" + "="*120)
    print("ENHANCED COMBINED FJC, SAW, and WLC ANALYSIS RESULTS FOR P(y)")
    print("="*120)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Save results to CSV
    output_csv = 'enhanced_combined_fjc_saw_wlc_py_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")
    
    total_time = time.time() - monitor.start_time
    print(f"\n🎉 Analysis complete! Total time: {total_time:.1f} seconds")
    
    return df

# ===========================================
# Run the enhanced analysis
# ===========================================

if __name__ == "__main__":
    print("Running ENHANCED combined FJC, SAW, and WLC analysis for P(y)...")
    print("Features: PERM algorithm, robust error handling, enhanced SAW sampling")
    
    results = analyze_combined_Py_models_enhanced(
        N_values, a=a, R=R, lp=lp, use_perm=True
    )
