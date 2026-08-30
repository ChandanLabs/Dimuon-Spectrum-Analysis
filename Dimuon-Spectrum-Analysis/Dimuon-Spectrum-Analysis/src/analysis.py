"""
analysis.py
───────────
Generates all figures for the Dimuon Spectrum Analysis project.
Run: python -m src.analysis   (from project root)
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import LogLocator, AutoMinorLocator

from src.physics_utils import (
    compute_invariant_mass, fit_resonance,
    gaussian_plus_background, compute_significance, PDG
)

# ── Styling ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

COLORS = {
    'data':      '#2c7bb6',
    'fit':       '#d7191c',
    'signal':    '#1a9641',
    'bkg':       '#fdae61',
    'jpsi':      '#7b2d8b',
    'upsilon':   '#e6692b',
    'z':         '#1a6faf',
    'vline':     '#555555',
}

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load & prepare data ────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'dimuon.csv'))
    df['M'] = compute_invariant_mass(
        df.E1, df.px1, df.py1, df.pz1,
        df.E2, df.px2, df.py2, df.pz2
    )
    # Quality cut: physical mass range
    df = df[(df.M > 1.0) & (df.M < 130.0)].copy()
    print(f"[Data] {len(df):,} events after quality cuts")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full spectrum (log scale)
# ══════════════════════════════════════════════════════════════════════════════
def plot_full_spectrum(df, save=True):
    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)

    ax_main = fig.add_subplot(gs[0])
    ax_res  = fig.add_subplot(gs[1], sharex=ax_main)

    mass = df['M'].values
    bins = np.linspace(1, 130, 600)
    counts, edges = np.histogram(mass, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_main.step(centers, counts, where='mid',
                 color=COLORS['data'], lw=0.8, label='CMS Dimuon Events (simulated)')
    ax_main.fill_between(centers, counts, step='mid',
                         color=COLORS['data'], alpha=0.15)

    ax_main.set_yscale('log')
    ax_main.set_ylabel('Event count', fontsize=12)
    ax_main.set_title('Dimuon Invariant Mass Spectrum  —  CMS Open Data (Simulated)',
                       fontsize=14, fontweight='bold', pad=12)
    ax_main.legend(fontsize=10, framealpha=0.7)

    # Reference lines + labels (after log scale so ylim is correct)
    resonances = [
        (PDG['J/psi'],   'J/ψ\n3.097 GeV',     COLORS['jpsi'],     'left'),
        (PDG['Upsilon'], 'Υ(1S)\n9.460 GeV',    COLORS['upsilon'],  'left'),
        (PDG['Z'],       'Z boson\n91.19 GeV',  COLORS['z'],        'left'),
    ]
    ylo, yhi = ax_main.get_ylim()
    label_y  = yhi * 0.25          # sit near the top (log scale)
    for mass_val, label, col, ha in resonances:
        ax_main.axvline(mass_val, color=col, lw=1.4, ls='--', alpha=0.85)
        offset = +1.0 if ha == 'left' else -1.0
        ax_main.text(mass_val + offset, label_y, label, color=col,
                     fontsize=9, va='top', ha=ha, fontweight='bold',
                     bbox=dict(fc='white', alpha=0.55, pad=1, ec='none'))

    # Info box
    info = (f"Total events: {len(mass):,}\n"
            f"Mass range:  1 – 130 GeV/c²\n"
            f"Source: CERN Open Data (CMS)")
    ax_main.text(0.98, 0.97, info, transform=ax_main.transAxes,
                 fontsize=8.5, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8, ec='grey'))

    # Residual panel — running 10-bin smoothed background subtraction
    smooth_n = 15
    smooth = np.convolve(counts, np.ones(smooth_n)/smooth_n, mode='same')
    residuals = (counts - smooth) / np.where(smooth > 0, np.sqrt(smooth), 1)
    ax_res.bar(centers, residuals, width=(bins[1]-bins[0]),
               color=np.where(np.abs(residuals) > 2, COLORS['fit'], COLORS['data']),
               alpha=0.6)
    ax_res.axhline(0,  color='k', lw=0.8)
    ax_res.axhline(2,  color='grey', lw=0.6, ls='--')
    ax_res.axhline(-2, color='grey', lw=0.6, ls='--')
    ax_res.set_ylabel('Residual\n(σ)', fontsize=9)
    ax_res.set_ylim(-5, 8)
    ax_res.set_yticks([-2, 0, 2, 4])

    plt.setp(ax_main.get_xticklabels(), visible=False)
    ax_res.set_xlabel('Invariant Mass  M  [GeV/c²]', fontsize=12)
    ax_res.set_xlim(1, 130)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_full_spectrum.png')
    if save:
        fig.savefig(path, bbox_inches='tight', dpi=200)
        print(f"[Fig 1] Saved → {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Z boson peak & fit
# ══════════════════════════════════════════════════════════════════════════════
def plot_peak(df, name, peak_center, window_gev, color, pdg_val,
              xlabel_unit='GeV/c²', n_bins=120, save_prefix='fig2'):
    mass  = df['M'].values
    res   = fit_resonance(mass, peak_center, window_gev, n_bins=n_bins)

    mu,    dmu   = res['fitted_mass']
    sigma, dsig  = res['fitted_sigma']
    chi2r        = res['chi2_reduced']
    popt         = res['popt']
    bc           = res['bin_centers']
    bcts         = res['bin_counts']
    berr         = res['bin_errors']
    lo, hi       = res['window']

    x_fine = np.linspace(lo, hi, 2000)
    y_fit  = gaussian_plus_background(x_fine, *popt)
    A, mu_f, sig_f, a_bg, b_bg = popt
    y_bkg  = a_bg * x_fine + b_bg
    y_sig  = gaussian_plus_background(x_fine, A, mu_f, sig_f, 0, 0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(bc, bcts, yerr=berr, fmt='o', ms=3.5, color='#333333',
                lw=0.8, capsize=2, label='Data', zorder=5)
    ax.fill_between(x_fine, y_bkg, alpha=0.25, color=COLORS['bkg'], label='Background (linear)')
    ax.fill_between(x_fine, y_bkg, y_fit, alpha=0.30, color=color, label='Signal (Gaussian)')
    ax.plot(x_fine, y_fit, color=COLORS['fit'], lw=2.0,
            label='Gaussian + background fit')
    ax.axvline(pdg_val, color='#555', lw=1.2, ls=':', label=f'PDG {pdg_val:.3f} GeV')

    # Results box
    dev = abs(mu - pdg_val) / dmu
    txt = (f"Measured mass:  {mu:.4f} ± {dmu:.4f} GeV\n"
           f"Width (σ):      {sigma:.4f} ± {dsig:.4f} GeV\n"
           f"PDG value:      {pdg_val:.4f} GeV\n"
           f"Deviation:      {dev:.1f}σ\n"
           f"χ²/ndf:         {chi2r:.2f}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            fontsize=9.5, va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9, ec='grey'))

    ax.set_xlabel(f'Invariant Mass  M  [{xlabel_unit}]', fontsize=12)
    ax.set_ylabel('Event count / bin', fontsize=12)
    ax.set_title(f'{name} Resonance — Gaussian + Background Fit', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.7)
    ax.set_xlim(lo, hi)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'{save_prefix}_{name.replace("/","_").replace(" ","_")}.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    print(f"[{save_prefix}] {name} peak saved → {path}")
    return fig, res


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Summary panel (2×2)
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_panel(df, all_results, save=True):
    mass = df['M'].values
    fig  = plt.figure(figsize=(14, 10))
    gs   = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # ── [0,0] Full spectrum ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(1, 130, 500)
    counts, edges = np.histogram(mass, bins=bins)
    centers = 0.5*(edges[:-1]+edges[1:])
    ax0.step(centers, np.where(counts>0, counts, np.nan),
             where='mid', color=COLORS['data'], lw=0.9)
    ax0.fill_between(centers, counts, step='mid', color=COLORS['data'], alpha=0.12)
    ax0.set_yscale('log')
    for mval, lbl, col in [(PDG['J/psi'],'J/ψ',COLORS['jpsi']),
                            (PDG['Upsilon'],'Υ',COLORS['upsilon']),
                            (PDG['Z'],'Z',COLORS['z'])]:
        ax0.axvline(mval, color=col, lw=1.2, ls='--', alpha=0.85, label=lbl)
    ax0.set_xlabel('M [GeV/c²]', fontsize=10)
    ax0.set_ylabel('Events', fontsize=10)
    ax0.set_title('Full Spectrum', fontsize=11, fontweight='bold')
    ax0.legend(fontsize=8, framealpha=0.6)

    # ── [0,1] Z boson peak ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    res_z  = all_results['Z']
    bc, bcts, berr = res_z['bin_centers'], res_z['bin_counts'], res_z['bin_errors']
    lo, hi = res_z['window']
    x_f = np.linspace(lo, hi, 1000)
    ax1.errorbar(bc, bcts, yerr=berr, fmt='o', ms=3, color='#333',
                 lw=0.8, capsize=2, zorder=5)
    ax1.plot(x_f, gaussian_plus_background(x_f, *res_z['popt']),
             color=COLORS['fit'], lw=2)
    mu_z, dmu_z = res_z['fitted_mass']
    ax1.set_title(f'Z Boson  ({mu_z:.2f} ± {dmu_z:.3f} GeV)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('M [GeV/c²]', fontsize=10); ax1.set_ylabel('Events', fontsize=10)

    # ── [1,0] J/ψ peak ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    res_j  = all_results['J/psi']
    bc, bcts, berr = res_j['bin_centers'], res_j['bin_counts'], res_j['bin_errors']
    lo2, hi2 = res_j['window']
    x_f2 = np.linspace(lo2, hi2, 1000)
    ax2.errorbar(bc, bcts, yerr=berr, fmt='o', ms=3, color='#333',
                 lw=0.8, capsize=2, zorder=5)
    ax2.plot(x_f2, gaussian_plus_background(x_f2, *res_j['popt']),
             color=COLORS['jpsi'], lw=2)
    mu_j, dmu_j = res_j['fitted_mass']
    ax2.set_title(f'J/ψ  ({mu_j:.4f} ± {dmu_j:.4f} GeV)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('M [GeV/c²]', fontsize=10); ax2.set_ylabel('Events', fontsize=10)

    # ── [1,1] Results table ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    particles = ['J/ψ', 'Υ(1S)', 'Z boson']
    pdg_vals  = [PDG['J/psi'], PDG['Upsilon'], PDG['Z']]
    keys      = ['J/psi', 'Upsilon', 'Z']
    rows = []
    for p, pdg, k in zip(particles, pdg_vals, keys):
        r   = all_results[k]
        mu, dmu   = r['fitted_mass']
        sig, dsig = r['fitted_sigma']
        dev = abs(mu - pdg) / dmu
        sig_info  = compute_significance(mass, pdg, abs(pdg * 0.05))
        rows.append([p,
                     f'{pdg:.4f}',
                     f'{mu:.4f} ± {dmu:.4f}',
                     f'{dev:.1f}σ',
                     f'{sig_info["significance"]:.0f}σ'])

    col_labels = ['Particle', 'PDG (GeV)', 'Measured (GeV)', 'Deviation', 'Significance']
    tbl = ax3.table(cellText=rows, colLabels=col_labels,
                    cellLoc='center', loc='center',
                    bbox=[0, 0.15, 1, 0.75])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#2c7bb6')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4f8')
    ax3.set_title('Summary of Results', fontsize=11, fontweight='bold', pad=12)

    fig.suptitle('Dimuon Invariant Mass Analysis  —  CMS Open Data',
                 fontsize=14, fontweight='bold', y=0.98)

    path = os.path.join(FIG_DIR, 'fig5_summary_panel.png')
    if save:
        fig.savefig(path, bbox_inches='tight', dpi=200)
        print(f"[Fig 5] Summary panel saved → {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Statistical significance bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_significance(df, save=True):
    mass = df['M'].values
    configs = [
        ('J/ψ',   PDG['J/psi'],   0.25,  COLORS['jpsi']),
        ('Υ(1S)', PDG['Upsilon'], 0.50,  COLORS['upsilon']),
        ('Z',     PDG['Z'],       4.50,  COLORS['z']),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    names, sigvals, sbratios = [], [], []
    for name, center, window, col in configs:
        s = compute_significance(mass, center, window)
        names.append(name)
        sigvals.append(s['significance'])
        sbratios.append(s['S'] / max(s['B'], 1))

    ax = axes[0]
    bars = ax.bar(names, sigvals, color=[COLORS['jpsi'], COLORS['upsilon'], COLORS['z']],
                  width=0.5, edgecolor='white', lw=1.5)
    ax.axhline(5, color='red', lw=1.2, ls='--', label='5σ discovery threshold')
    for bar, val in zip(bars, sigvals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}σ', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Statistical Significance  (S/√B)', fontsize=11)
    ax.set_title('Peak Significance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(sigvals) * 1.25)

    ax2 = axes[1]
    bars2 = ax2.bar(names, sbratios,
                    color=[COLORS['jpsi'], COLORS['upsilon'], COLORS['z']],
                    width=0.5, edgecolor='white', lw=1.5)
    for bar, val in zip(bars2, sbratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Signal-to-Background Ratio  (S/B)', fontsize=11)
    ax2.set_title('Signal-to-Background Ratio', fontsize=12, fontweight='bold')

    fig.suptitle('Statistical Analysis of Particle Resonances', fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'fig6_significance.png')
    if save:
        fig.savefig(path, bbox_inches='tight', dpi=200)
        print(f"[Fig 6] Significance plot saved → {path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("═" * 60)
    print("  Dimuon Spectrum Analysis — CMS Open Data")
    print("═" * 60)

    df   = load_data()
    mass = df['M'].values

    print("\n[Step 1/4] Full spectrum figure ...")
    plot_full_spectrum(df)

    print("\n[Step 2/4] Individual resonance peaks ...")
    _, res_z   = plot_peak(df, 'Z',        91.1876, 12.0, COLORS['z'],
                            PDG['Z'],   n_bins=120, save_prefix='fig2')
    _, res_j   = plot_peak(df, 'J/psi',    3.0969,  0.6,  COLORS['jpsi'],
                            PDG['J/psi'],n_bins=120, save_prefix='fig3')
    _, res_ups = plot_peak(df, 'Upsilon',   9.4603,  1.2,  COLORS['upsilon'],
                            PDG['Upsilon'],n_bins=120, save_prefix='fig4')

    all_results = {'Z': res_z, 'J/psi': res_j, 'Upsilon': res_ups}

    print("\n[Step 3/4] Summary panel ...")
    plot_summary_panel(df, all_results)

    print("\n[Step 4/4] Significance plots ...")
    plot_significance(df)

    # ── Console report ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RESULTS SUMMARY")
    print("═" * 60)
    header = f"{'Particle':<10} {'PDG (GeV)':<12} {'Measured (GeV)':<22} {'Deviation':<12} {'χ²/ndf'}"
    print(header)
    print("-" * 70)
    pairs = [('J/ψ',    PDG['J/psi'],   res_j),
             ('Υ(1S)',  PDG['Upsilon'], res_ups),
             ('Z boson',PDG['Z'],       res_z)]
    for name, pdg, r in pairs:
        mu, dmu   = r['fitted_mass']
        sig, dsig = r['fitted_sigma']
        dev       = abs(mu - pdg) / dmu
        print(f"{name:<10} {pdg:<12.4f} {mu:.4f} ± {dmu:.4f}      "
              f"{dev:<12.1f} {r['chi2_reduced']:.2f}")
    print("═" * 60)
    print(f"All figures saved to  ./figures/")


if __name__ == '__main__':
    main()
