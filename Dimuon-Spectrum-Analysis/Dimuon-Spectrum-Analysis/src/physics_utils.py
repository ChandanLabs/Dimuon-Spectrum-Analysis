"""
physics_utils.py
────────────────
Utility functions for dimuon invariant mass analysis.
All energies and momenta are in GeV / GeV/c. Masses in GeV/c².
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist


# ── PDG reference values (GeV/c²) ─────────────────────────────────────────────
PDG = {
    "J/psi": 3.0969,
    "Upsilon": 9.4603,
    "Z":    91.1876,
}


def compute_invariant_mass(E1, px1, py1, pz1, E2, px2, py2, pz2):
    """
    Reconstruct the invariant mass of a dimuon pair from 4-momenta.

    Uses the Minkowski metric signature (+,-,-,-):
        M² = (E1+E2)² − (px1+px2)² − (py1+py2)² − (pz1+pz2)²

    Parameters
    ----------
    E1, E2   : float or array-like — energy of each muon [GeV]
    px1, px2 : float or array-like — x-momentum [GeV/c]
    py1, py2 : float or array-like — y-momentum [GeV/c]
    pz1, pz2 : float or array-like — z-momentum [GeV/c]

    Returns
    -------
    M : ndarray — invariant mass [GeV/c²]; negative M² clipped to 0.
    """
    E_sum  = np.asarray(E1)  + np.asarray(E2)
    px_sum = np.asarray(px1) + np.asarray(px2)
    py_sum = np.asarray(py1) + np.asarray(py2)
    pz_sum = np.asarray(pz1) + np.asarray(pz2)
    M2 = E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
    return np.sqrt(np.clip(M2, 0, None))


def gaussian(x, amplitude, mean, sigma):
    """
    Normalised Gaussian function.

        f(x) = amplitude · exp(−½·((x−mean)/sigma)²)

    Parameters
    ----------
    x         : array-like — evaluation points
    amplitude : float      — peak height
    mean      : float      — peak centre [GeV/c²]
    sigma     : float      — standard deviation (≈ detector resolution + natural width)

    Returns
    -------
    ndarray — Gaussian values at x
    """
    return amplitude * np.exp(-0.5 * ((np.asarray(x) - mean) / sigma) ** 2)


def gaussian_plus_background(x, A, mu, sigma, a, b):
    """
    Gaussian resonance peak superimposed on a linear background.

        f(x) = A·exp(−½·((x−μ)/σ)²) + a·x + b

    The linear background models the Drell-Yan continuum underneath each
    resonance peak.

    Parameters
    ----------
    x     : array-like — invariant mass values [GeV/c²]
    A     : float      — signal peak amplitude
    mu    : float      — resonance mass [GeV/c²]
    sigma : float      — peak width [GeV/c²]
    a     : float      — background slope
    b     : float      — background intercept

    Returns
    -------
    ndarray — model values at x
    """
    x = np.asarray(x, dtype=float)
    return gaussian(x, A, mu, sigma) + a * x + b


def fit_resonance(mass_array, peak_center, window_gev, p0=None, n_bins=200):
    """
    Fit a Gaussian + linear-background model to a resonance peak.

    Parameters
    ----------
    mass_array  : array-like — full invariant mass array [GeV/c²]
    peak_center : float      — expected resonance mass [GeV/c²]
    window_gev  : float      — ± half-width of fit window [GeV]
    p0          : list, optional — initial parameter guess [A, mu, sigma, a, b]
    n_bins      : int        — number of histogram bins in the window

    Returns
    -------
    dict with keys
        popt          : best-fit parameters [A, mu, sigma, a, b]
        pcov          : covariance matrix
        perr          : 1-σ parameter uncertainties
        fitted_mass   : μ ± δμ [GeV/c²]
        fitted_sigma  : σ ± δσ [GeV/c²]
        chi2_reduced  : χ²/ndf
        bin_centers   : bin centre values for plotting
        bin_counts    : histogram counts
        bin_errors    : Poisson uncertainties on counts
    """
    mass_array = np.asarray(mass_array)
    lo = peak_center - window_gev
    hi = peak_center + window_gev
    mask = (mass_array >= lo) & (mass_array <= hi)
    window_mass = mass_array[mask]

    if len(window_mass) < 20:
        raise ValueError(f"Too few events ({len(window_mass)}) in window "
                         f"[{lo:.2f}, {hi:.2f}] GeV")

    counts, edges = np.histogram(window_mass, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    errors  = np.where(counts > 0, np.sqrt(counts), 1.0)   # Poisson σ

    # Default initial guess
    if p0 is None:
        A0    = counts.max()
        mu0   = peak_center
        sig0  = window_gev / 4
        # Rough background from edges
        bkg   = 0.5 * (counts[:5].mean() + counts[-5:].mean())
        a0    = (counts[-5:].mean() - counts[:5].mean()) / (hi - lo)
        b0    = bkg - a0 * mu0
        p0 = [A0, mu0, sig0, a0, b0]

    bounds_lo = [0,    lo,  1e-4, -np.inf, -np.inf]
    bounds_hi = [np.inf, hi, window_gev, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(
            gaussian_plus_background, centers, counts,
            p0=p0, sigma=errors, absolute_sigma=True,
            bounds=(bounds_lo, bounds_hi), maxfev=20_000
        )
    except RuntimeError as exc:
        raise ValueError(f"Curve fit did not converge: {exc}") from exc

    perr = np.sqrt(np.diag(pcov))
    A, mu, sigma, a, b = popt
    dA, dmu, dsigma, da, db = perr

    # χ² / ndf
    fitted_vals = gaussian_plus_background(centers, *popt)
    residuals   = (counts - fitted_vals) / errors
    chi2_val    = np.sum(residuals ** 2)
    ndf         = (counts > 0).sum() - len(popt)
    chi2_red    = chi2_val / max(ndf, 1)

    return {
        "popt":         popt,
        "pcov":         pcov,
        "perr":         perr,
        "fitted_mass":  (mu, dmu),
        "fitted_sigma": (abs(sigma), dsigma),
        "chi2_reduced": chi2_red,
        "chi2":         chi2_val,
        "ndf":          ndf,
        "bin_centers":  centers,
        "bin_counts":   counts,
        "bin_errors":   errors,
        "window":       (lo, hi),
        "p_value":      1 - chi2_dist.cdf(chi2_val, ndf),
    }


def compute_significance(mass_array, peak_center, window_gev):
    """
    Estimate signal significance S / √B for a resonance peak.

    Splits the window into a narrow signal band (±2σ_est around peak)
    and side-band regions for background estimation.

    Parameters
    ----------
    mass_array  : array-like — full mass spectrum [GeV/c²]
    peak_center : float      — resonance position [GeV/c²]
    window_gev  : float      — total fit window half-width [GeV]

    Returns
    -------
    dict with keys: S (signal events), B (background estimate),
                    significance (S/√B), total_in_window
    """
    mass_array = np.asarray(mass_array)
    lo = peak_center - window_gev
    hi = peak_center + window_gev
    sigma_est = window_gev / 3.0           # rough width estimate

    sig_lo = peak_center - 2 * sigma_est
    sig_hi = peak_center + 2 * sigma_est

    n_total  = ((mass_array >= lo)     & (mass_array <= hi)    ).sum()
    n_signal = ((mass_array >= sig_lo) & (mass_array <= sig_hi)).sum()
    n_side   = n_total - n_signal

    # Scale sideband to signal-region width
    sig_width  = sig_hi  - sig_lo
    side_width = (hi - lo) - sig_width
    B = n_side * sig_width / side_width if side_width > 0 else 0
    S = max(n_signal - B, 0)
    significance = S / np.sqrt(max(B, 1))

    return {
        "S":          S,
        "B":          B,
        "significance": significance,
        "total_in_window": n_total,
    }
