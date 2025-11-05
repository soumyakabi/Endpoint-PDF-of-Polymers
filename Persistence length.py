# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 19:30:37 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from numba import njit
import pandas as pd

# ============= PARAMETERS =============
a = 0.1          # Kuhn length (μm)
L = 2.0          # Confinement length (μm)
x0 = 0.75        # Tethering position (μm)
lp_values = [0.005, 0.025, 0.1, 0.3]
N0 = 25          # Fixed N for lp scan
fjc_walkers = 200000
wlc_walkers = 50000

# ============ ANALYTICAL FOURIER SOLUTION ============
def analytical_Px_fourier(x_vals, x0, a, N, L, n_terms=200):
    D = a**2 / 2
    G = np.zeros_like(x_vals)
    for n in range(1, n_terms):
        sin_nx0 = np.sin(n * np.pi * x0 / L)
        sin_nx = np.sin(n * np.pi * x_vals / L)
        decay = np.exp(-((n * np.pi / L)**2) * D * N)
        G += sin_nx0 * sin_nx * decay
    G *= (2 / L)
    norm = trapezoid(G, x_vals)
    return G / norm if norm > 0 else G

# ============ FJC MONTE CARLO ============
def fjc_monte_carlo_Px(x0, a, N, L, n_walkers, seed):
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(n_walkers):
        x = x0; alive = True
        for _ in range(N):
            x += a * rng.normal()
            if x < 0 or x > L:
                alive = False; break
        if alive: finals.append(x)
    return np.array(finals)

# ============ WLC MONTE CARLO (Numba-robust) ============
@njit
def generate_wlc_chain(N, a, lp, x0, L):
    coords = np.zeros((N+1, 3))
    coords[0,0] = x0
    t0 = np.random.normal(0,1)
    t1 = np.random.normal(0,1)
    t2 = np.random.normal(0,1)
    norm = np.sqrt(t0*t0 + t1*t1 + t2*t2)
    t = np.array([t0, t1, t2]) / norm
    for i in range(1, N+1):
        coords[i] = coords[i-1] + a * t
        if coords[i,0] < 0 or coords[i,0] > L: return -1.0
        if i<N:
            sigma = np.sqrt(a/lp) if lp > 1e-8 else np.pi
            theta = np.random.normal(0, sigma)
            vx = np.random.normal(0, 1)
            vy = np.random.normal(0, 1)
            vz = np.random.normal(0, 1)
            v = np.array([vx, vy, vz])
            k = v - np.dot(v, t) * t
            knorm = np.sqrt(np.sum(k*k))
            if knorm > 1e-8:
                k /= knorm
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                t = t*cos_theta + np.cross(k, t)*sin_theta + k*np.dot(k, t)*(1-cos_theta)
                t /= np.sqrt(np.sum(t*t))
    return coords[N,0]

def wlc_monte_carlo_Px(x0, a, N, L, lp, n_walkers, seed):
    np.random.seed(seed)
    finals = []
    for _ in range(n_walkers):
        x = generate_wlc_chain(N, a, lp, x0, L)
        if x>=0: finals.append(x)
    return np.array(finals)

# ============ METRICS & THEORETICAL WLC ============
def calculate_kl_chi2(data, model_pdf, x_vals, bins):
    hist, edges = np.histogram(data, bins=bins, range=(0,x_vals[-1]), density=False)
    centers = 0.5*(edges[:-1]+edges[1:]); bw=edges[1]-edges[0]
    total = hist.sum()
    hist_prob = hist/total
    model_vals = np.interp(centers,x_vals,model_pdf)
    model_prob = model_vals*bw
    model_prob/=model_prob.sum()
    mask = (hist_prob>0)&(model_prob>0)
    kl = np.sum(hist_prob[mask]*np.log(hist_prob[mask]/model_prob[mask]))
    exp_counts = model_prob*total
    o=hist[mask]; e=exp_counts[mask]
    if np.all(e>5):
        χ2 = np.sum((o-e)**2/e)/(len(o)-1)
    else:
        χ2=np.nan
    return kl,χ2

def theoretical_wlc_variance(N,a,lp):
    if lp<1e-6: return N*a*a/3
    return N*a*a/3*(1+2*lp/a)/(1+lp/a)

# ============ MAIN: lp DEPENDENCE ============
def lp_dependence_Px():
    x_vals = np.linspace(0, L, 400)
    analytic = analytical_Px_fourier(x_vals, x0, a, N0, L)
    rows_lp = []
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    for i, lp in enumerate(lp_values):
        ax = axs.flatten()[i]
        # FJC
        fjc = fjc_monte_carlo_Px(x0, a, N0, L, fjc_walkers, seed=100+i)
        # WLC
        wlc = wlc_monte_carlo_Px(x0, a, N0, L, lp, wlc_walkers, seed=200+i)
        kl, χ2 = calculate_kl_chi2(wlc, analytic, x_vals, bins=60)
        # Additional statistics
        fjc_mean = np.mean(fjc); fjc_std = np.std(fjc)
        wlc_mean = np.mean(wlc); wlc_std = np.std(wlc)
        mean_diff = abs(wlc_mean - fjc_mean)
        std_ratio = wlc_std / fjc_std if fjc_std>0 else np.nan
        # Regime label
        if lp/a < 0.1:
            expected = "≈ FJC (flexible)"
        elif lp/a < 1:
            expected = "Semi-flexible"
        else:
            expected = "Stiff/rod-like"
        rows_lp.append({'lp':lp, 'lp/a':lp/a, 'Samples':len(wlc),
                        'Mean_Diff':round(mean_diff,3), 'Std_Ratio':round(std_ratio,3),
                        'Expected':expected, 'kl_wlc':kl, 'chi2_wlc':χ2,
                        'wlc_mean':wlc_mean, 'wlc_std':wlc_std})
        ax.hist(fjc, bins=60, range=(0,L), density=True, alpha=0.4, label='FJC MC')
        ax.hist(wlc, bins=60, range=(0,L), density=True, alpha=0.8, color="#cc5500",
                edgecolor='black', linewidth=0.5, label=f'WLC MC (lp={lp})')
        ax.plot(x_vals, analytic, 'k-', linewidth=2, label='Analytical')
        ax.set_title(rf'$l_p$ = {lp} $\mu$m ($l_p/a$={lp/a:.3f})', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_xlabel('x (μm)', fontsize=18)
        ax.set_ylabel('P(x)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig('Px_vs_lp_only.pdf', dpi=300)
    pd.DataFrame(rows_lp).to_csv('metrics_vs_lp_only.csv', index=False)
    plt.show()

if __name__ == '__main__':
    lp_dependence_Px()
