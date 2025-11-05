# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:32:16 2025

@author: SOUMYA
"""

# ===========================================
# Tethering Point Sensitivity (improved bootstrap shading)
# FJC (hist), SAW (KDE + bootstrap band), WLC (KDE) vs Fourier analytic
# ===========================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from numba import njit
import pandas as pd
import os

# ------------------------
# User parameters
# ------------------------
a = 0.1
L = 2.0
N_fixed = 25
x0_values = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
lp = 0.2

# Sim sizes (adjust for runtime)
n_walkers_fjc = 200_000
n_walkers_saw = 3000
n_walkers_wlc = 50_000

# Plot / bootstrap params
x_grid = np.linspace(0, L, 401)
bootstrap_K = 200               # number of bootstrap KDE draws
min_samples_for_kde_boot = 30   # if fewer, fallback to histogram
saw_kde_bw_base = 1.2
output_dir = "."
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# Analytic Fourier P(x)
# ------------------------
def analytical_Px_fourier(x_vals, x0, a, N, L, n_terms=1000, decay_tol=1e-12):
    D = a**2 / 2.0
    G = np.zeros_like(x_vals)
    for n in range(1, n_terms):
        decay = np.exp(- (n * np.pi / L)**2 * D * N)
        if decay < decay_tol:
            break
        G += np.sin(n * np.pi * x0 / L) * np.sin(n * np.pi * x_vals / L) * decay
    G *= (2.0 / L)
    norm = trapezoid(G, x_vals)
    return (G / norm) if (norm > 0) else np.ones_like(x_vals) / L

# ------------------------
# FJC Monte Carlo (1D gaussian steps, absorbing)
# ------------------------
def fjc_monte_carlo_Px(x0, a, N, L, n_walkers=200000, seed=42):
    rng = np.random.default_rng(seed)
    ends = []
    for _ in range(n_walkers):
        x = x0
        alive = True
        for _ in range(N):
            x += a * rng.normal()
            if (x < 0) or (x > L):
                alive = False
                break
        if alive:
            ends.append(x)
    return np.array(ends)

# ------------------------
# SAW Monte Carlo (off-lattice 3D) -- Numba
# ------------------------
@njit
def generate_saw_chain(N, a, x0, L, bead_radius):
    coords = np.zeros((N+1, 3))
    coords[0, 0] = x0
    for i in range(1, N+1):
        step = np.random.normal(0.0, 1.0, 3)
        norm_s = np.sqrt(np.sum(step**2))
        if norm_s <= 1e-12:
            return -1.0
        step = step / norm_s
        coords[i] = coords[i-1] + a * step
        # absorbing
        if coords[i,0] < 0.0 or coords[i,0] > L:
            return -1.0
        # excluded volume
        for j in range(i):
            dx = coords[i] - coords[j]
            if np.sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]) < 2.0*bead_radius:
                return -1.0
    return coords[-1,0]

def saw_monte_carlo_Px_with_diagnostics(x0, a, N, L, bead_radius=0.03, target_accepted=3000, max_attempts_factor=500):
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * target_accepted * (N//10 + 1)
    while len(ends) < target_accepted and attempts < max_attempts:
        val = generate_saw_chain(N, a, x0, L, bead_radius)
        if val >= 0.0:
            ends.append(val)
        attempts += 1
    accepted = len(ends)
    accept_rate = accepted / attempts if attempts>0 else 0.0
    return np.array(ends), accepted, attempts, accept_rate

# ------------------------
# WLC Monte Carlo (correlated tangents) -- Numba
# ------------------------
@njit
def generate_wlc_chain(N, a, lp, x0, L):
    coords = np.zeros((N+1, 3))
    tangents = np.zeros((N+1, 3))
    coords[0,0] = x0
    t0 = np.random.normal(0.0, 1.0, 3)
    norm_t0 = np.sqrt(np.sum(t0*t0))
    if norm_t0 <= 1e-12:
        return -1.0
    t0 /= norm_t0
    tangents[0] = t0
    alpha = np.exp(-a / lp)
    sqrt_term = np.sqrt(max(0.0, 1.0 - alpha*alpha))
    for i in range(1, N+1):
        v = np.random.normal(0.0, 1.0, 3)
        nv = np.sqrt(np.sum(v*v))
        if nv <= 1e-12:
            return -1.0
        v /= nv
        new_t = alpha * tangents[i-1] + sqrt_term * v
        nt_norm = np.sqrt(np.sum(new_t*new_t))
        if nt_norm <= 1e-12:
            return -1.0
        new_t /= nt_norm
        tangents[i] = new_t
        coords[i] = coords[i-1] + a * new_t
        if coords[i,0] < 0.0 or coords[i,0] > L:
            return -1.0
    return coords[-1,0]

def wlc_monte_carlo_Px(x0, a, N, L, lp=0.2, n_walkers=50000, max_attempts_factor=200):
    ends = []
    attempts = 0
    max_attempts = max_attempts_factor * n_walkers
    while len(ends) < n_walkers and attempts < max_attempts:
        val = generate_wlc_chain(N, a, lp, x0, L)
        if val >= 0.0:
            ends.append(val)
        attempts += 1
    return np.array(ends)

# ------------------------
# Absorbing-consistent KDE (renormalized on [0,L])
# ------------------------
def absorbing_kde_on_grid(data, x_grid, bw_factor=1.2, L=2.0):
    if len(data) < 3:
        return None
    kde = gaussian_kde(data)
    kde.set_bandwidth(kde.factor * bw_factor)
    pdf = kde(x_grid)
    pdf = np.clip(pdf, 0.0, None)
    mask = (x_grid >= 0.0) & (x_grid <= L)
    norm = trapezoid(pdf[mask], x_grid[mask])
    if norm <= 0 or not np.isfinite(norm):
        return None
    return pdf / norm

# ------------------------
# Bootstrap KDEs -> mean PDF and ±1σ band
# ------------------------
def bootstrap_kde_band(data, x_grid, N_for_bw, n_boot=200, bw_base=1.2, L=2.0):
    """Compute bootstrap KDE mean and std on x_grid (absorbing-renormalized)"""
    if len(data) < min_samples_for_kde_boot:
        return None, None, None  # signal insufficient samples for reliable KDE bootstraps
    pdfs = []
    # adaptive bw: larger smoothing for small N (to reduce wiggles)
    bw_factor = bw_base + 20.0 / max(1.0, float(N_for_bw))
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        pdf = absorbing_kde_on_grid(sample, x_grid, bw_factor=bw_factor, L=L)
        if pdf is not None:
            pdfs.append(pdf)
    if len(pdfs) == 0:
        return None, None, None
    arr = np.vstack(pdfs)
    mean_pdf = np.mean(arr, axis=0)
    std_pdf = np.std(arr, axis=0)
    return mean_pdf, mean_pdf - std_pdf, mean_pdf + std_pdf

# ------------------------
# Fit metrics (KL & reduced chi^2)
# ------------------------
def calculate_fit_metrics(data, expected_pdf, x_grid, L=2.0, n_bins=50):
    if len(data) < 20:
        return np.nan, np.nan
    hist_obs, edges = np.histogram(data, bins=n_bins, range=(0, L), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    expected_vals = np.interp(centers, x_grid, expected_pdf)
    p_obs = hist_obs * width
    p_exp = expected_vals * width
    # normalize
    if np.sum(p_exp) <= 0 or np.sum(p_obs) <= 0:
        return np.nan, np.nan
    p_obs /= np.sum(p_obs)
    p_exp /= np.sum(p_exp)
    eps = 1e-12
    p_obs = np.clip(p_obs, eps, None)
    p_exp = np.clip(p_exp, eps, None)
    KL = np.sum(p_obs * np.log(p_obs / p_exp))
    counts_obs, _ = np.histogram(data, bins=n_bins, range=(0, L))
    expected_counts = expected_vals * len(data) * width
    mask = expected_counts > 1.0
    if np.sum(mask) < 5:
        redchi = np.nan
    else:
        chi2 = np.sum((counts_obs[mask] - expected_counts[mask])**2 / expected_counts[mask])
        dof = np.sum(mask) - 1
        redchi = chi2 / dof if dof > 0 else np.nan
    return KL, redchi

# ------------------------
# Main analysis & plotting (improved bootstrap shading)
# ------------------------
def analyze_and_plot(x0_values, N=25, a=0.1, L=2.0, lp=0.2,
                     n_walkers_fjc=200000, n_walkers_saw=3000, n_walkers_wlc=50000,
                     bootstrap_K=200, output_dir="."):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    results = []
    for idx, x0 in enumerate(x0_values):
        ax = axes[idx]
        print(f"Processing x0 = {x0} ...")
        # simulate
        fjc_data = fjc_monte_carlo_Px(x0, a, N, L, n_walkers=n_walkers_fjc, seed=42)
        saw_data, accepted, attempts, accept_rate = saw_monte_carlo_Px_with_diagnostics(x0, a, N, L,
                                                                                       bead_radius=0.03,
                                                                                       target_accepted=n_walkers_saw,
                                                                                       max_attempts_factor=500)
        wlc_data = wlc_monte_carlo_Px(x0, a, N, L, lp=lp, n_walkers=n_walkers_wlc)
        fourier_pdf = analytical_Px_fourier(x_grid, x0, a, N, L)
        # Plot FJC histogram (back)
        ax.hist(fjc_data, bins=120, range=(0, L), density=True,
                alpha=0.45, color="cornflowerblue", edgecolor="black", linewidth=0.25, zorder=1,
                label="FJC MC (hist)")
        # SAW: try bootstrap KDE band if enough accepted samples
        if len(saw_data) >= min_samples_for_kde_boot:
            mean_pdf, low_pdf, high_pdf = bootstrap_kde_band(saw_data, x_grid, N_for_bw=N,
                                                             n_boot=bootstrap_K, bw_base=saw_kde_bw_base, L=L)
            if mean_pdf is not None:
                # band
                ax.fill_between(x_grid, low_pdf, high_pdf, color="green", alpha=0.35, zorder=2,
                                label="SAW ±1σ (bootstrap KDE)")
                # mean
                ax.plot(x_grid, mean_pdf, color="green", lw=2.5, zorder=3, label="SAW MC (mean KDE)")
        else:
            # fallback: histogram + bootstrap hist shading (less smooth)
            centers, mean_hist, std_hist = bootstrap_hist(saw_data, bins=40, n_boot=300, L=L)
            if centers is not None:
                ax.bar(centers, mean_hist, width=L/40, alpha=0.5, color="green", edgecolor="none", zorder=2,
                       label=f"SAW MC (hist, n={len(saw_data)})")
                ax.fill_between(centers, mean_hist - std_hist, mean_hist + std_hist, color="green", alpha=0.25, zorder=2)
        # WLC: KDE on grid if possible (and small bootstrap shading optional)
        if len(wlc_data) >= 20:
            # compute single KDE (optionally could compute bootstrap too)
            wlc_pdf = absorbing_kde_on_grid(wlc_data, x_grid, bw_factor=1.2, L=L)
            if wlc_pdf is not None:
                ax.plot(x_grid, wlc_pdf, color="darkorange", lw=2.0, zorder=3, label="WLC MC (KDE)")
        else:
            centers_w, mean_w, std_w = bootstrap_hist(wlc_data, bins=40, n_boot=200, L=L)
            if centers_w is not None:
                ax.bar(centers_w, mean_w, width=L/40, alpha=0.5, color="darkorange", edgecolor="none", zorder=2,
                       label=f"WLC MC (hist, n={len(wlc_data)})")
        # Fourier analytic on top
        ax.plot(x_grid, fourier_pdf, color="black", lw=2.5, zorder=4, label="Fourier analytic")
        # formatting
        ax.set_title(f"x0 = {x0:.2f} μm, N = {N}", fontsize=16, fontweight="bold")
        ax.set_xlim(0, L); ax.set_ylim(0, 2.5)
        ax.set_xlabel("x (μm)", fontsize=18, fontweight="bold")
        ax.set_ylabel("P(x)", fontsize=18, fontweight="bold")
        ax.grid(alpha=0.25)
        for t in (ax.get_xticklabels() + ax.get_yticklabels()):
            t.set_fontsize(14); t.set_fontweight("bold")
        if idx == 0:
            ax.legend(fontsize=10)
        # annotate acceptance
        ax.text(0.98, 0.95, f"SAW accepted: {accepted}\naccept rate: {accept_rate:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8), zorder=5)
        # metrics
        KL_saw, chi2_saw = calculate_fit_metrics(saw_data, fourier_pdf, x_grid, L=L)
        KL_fjc, chi2_fjc = calculate_fit_metrics(fjc_data, fourier_pdf, x_grid, L=L)
        KL_wlc, chi2_wlc = calculate_fit_metrics(wlc_data, fourier_pdf, x_grid, L=L)
        results.append({"x0": x0, "N": N, "SAW_n": len(saw_data), "SAW_accepted": accepted,
                        "SAW_attempts": attempts, "SAW_accept_rate": accept_rate,
                        "KL_SAW": KL_saw, "Chi2_SAW": chi2_saw,
                        "KL_FJC": KL_fjc, "Chi2_FJC": chi2_fjc,
                        "KL_WLC": KL_wlc, "Chi2_WLC": chi2_wlc})
    # finalize
    for j in range(len(x0_values), len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    out_pdf = os.path.join(output_dir, "tethering_sensitivity_bootstrap_kde.pdf")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print("Saved figure to", out_pdf)
    df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, "tethering_sensitivity_results_bootstrap_kde.csv")
    df.to_csv(out_csv, index=False)
    print("Saved metrics to", out_csv)
    plt.show()
    return df

# ------------------------
# Run (example)
# ------------------------
if __name__ == "__main__":
    df = analyze_and_plot(x0_values, N=N_fixed, a=a, L=L, lp=lp,
                          n_walkers_fjc=n_walkers_fjc, n_walkers_saw=n_walkers_saw,
                          n_walkers_wlc=n_walkers_wlc, bootstrap_K=bootstrap_K, output_dir=output_dir)
    print(df)
