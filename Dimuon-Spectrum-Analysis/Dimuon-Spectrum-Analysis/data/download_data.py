"""
download_data.py
────────────────
Attempts to download the CMS dimuon dataset from CERN Open Data Portal.
Falls back to physics-accurate synthetic generation if the portal is unavailable.

Usage:  python download_data.py
"""

import os
import sys
import numpy as np
import pandas as pd

OUTPUT = os.path.join(os.path.dirname(__file__), 'dimuon.csv')

SOURCES = [
    ("CERN Open Data Portal (primary)",
     "http://opendata.cern.ch/record/700/files/Dimuon_DoubleMu.csv"),
    ("CERN Open Data Portal (mirror)",
     "https://opendata.cern.ch/record/700/files/Dimuon_DoubleMu.csv"),
]


def try_download():
    """Try each URL source in order. Return True if any succeeds."""
    try:
        import urllib.request
        for name, url in SOURCES:
            try:
                print(f"  Trying: {name} ...")
                urllib.request.urlretrieve(url, OUTPUT)
                lines = sum(1 for _ in open(OUTPUT)) - 1
                if lines > 100:
                    print(f"  ✓ Downloaded {lines:,} events from {name}")
                    return True
                else:
                    print(f"  ✗ File too small ({lines} lines), skipping")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    except ImportError:
        pass
    return False


def generate_synthetic():
    """
    Generate a physics-accurate synthetic dimuon dataset.

    Uses proper Lorentz boosts so that the invariant mass reconstructed
    from the 4-momentum components exactly matches the generated parent mass.
    Resonances (J/ψ, Υ, Z) are sampled from Gaussian approximations of
    Breit-Wigner line shapes with PDG-accurate masses and widths.
    """
    print("  Generating physics-accurate synthetic dataset ...")
    rng = np.random.default_rng(42)
    N   = 100_000

    # Component fractions and resonance parameters
    components = [
        ("Z",        91.1876,  2.495/2,  0.14, 25.0),
        ("Upsilon",   9.4603,  0.054,    0.04,  6.0),
        ("J/psi",     3.0969,  0.093,    0.06,  5.0),
    ]
    W_BKG = 1.0 - sum(c[3] for c in components)

    def gen_4vecs(masses, pt_mean=20.0, eta_max=2.4):
        """
        Generate muon-pair 4-vectors from parent masses via proper Lorentz boost.
        Invariant mass is exactly conserved by construction.
        """
        M   = np.asarray(masses, dtype=float)
        n   = len(M)
        pT  = rng.exponential(pt_mean, n)
        phi = rng.uniform(-np.pi, np.pi, n)
        eta = rng.uniform(-eta_max, eta_max, n)
        px_p, py_p = pT * np.cos(phi), pT * np.sin(phi)
        pz_p = pT * np.sinh(eta)
        E_p  = np.sqrt(M**2 + px_p**2 + py_p**2 + pz_p**2)

        bx, by, bz = px_p/E_p, py_p/E_p, pz_p/E_p
        beta2  = bx**2 + by**2 + bz**2
        gamma  = E_p / M

        cos_th = rng.uniform(-1, 1, n)
        sin_th = np.sqrt(np.clip(1 - cos_th**2, 0, None))
        phi_d  = rng.uniform(-np.pi, np.pi, n)
        p_star = M / 2

        px1c = p_star * sin_th * np.cos(phi_d)
        py1c = p_star * sin_th * np.sin(phi_d)
        pz1c = p_star * cos_th

        def boost(Ec, pxc, pyc, pzc):
            safe_b2 = np.where(beta2 > 1e-12, beta2, 1e-12)
            bdotp   = bx*pxc + by*pyc + bz*pzc
            coeff   = (gamma - 1) / safe_b2 * bdotp + gamma * Ec
            return (gamma*(Ec + bdotp),
                    pxc + bx*coeff,
                    pyc + by*coeff,
                    pzc + bz*coeff)

        E1, px1, py1, pz1 = boost(p_star,  px1c,  py1c,  pz1c)
        E2, px2, py2, pz2 = boost(p_star, -px1c, -py1c, -pz1c)
        return E1, px1, py1, pz1, E2, px2, py2, pz2

    arrs = {k: [] for k in ['E1','px1','py1','pz1','E2','px2','py2','pz2']}

    for name, mass, sigma, frac, pt_mean in components:
        n = int(N * frac)
        m = rng.normal(mass, sigma, n)
        vecs = gen_4vecs(m, pt_mean=pt_mean)
        for key, arr in zip(arrs.keys(), vecs):
            arrs[key].append(arr)

    # Background: falling Drell-Yan continuum
    n_bkg = N - sum(int(N*c[3]) for c in components)
    tau   = 18.0
    u     = rng.uniform(0, 1, n_bkg)
    m_bkg = -tau * np.log(1 - u*(1 - np.exp(-129.0/tau))) + 1.0
    vecs  = gen_4vecs(m_bkg, pt_mean=15.0)
    for key, arr in zip(arrs.keys(), vecs):
        arrs[key].append(arr)

    idx = rng.permutation(N)
    df  = pd.DataFrame({k: np.concatenate(v)[idx] for k, v in arrs.items()})
    df.insert(0, 'Type',  'GT')
    df.insert(1, 'Run',   rng.integers(100000, 999999, N)[idx])
    df.insert(2, 'Event', rng.integers(1, 10**7,  N)[idx])
    df['Q1'] = rng.choice([-1, 1], N)[idx]
    df['Q2'] = rng.choice([-1, 1], N)[idx]

    df.to_csv(OUTPUT, index=False)
    print(f"  ✓ Synthetic dataset: {len(df):,} events saved to {OUTPUT}")


if __name__ == '__main__':
    print("─" * 55)
    print("  CMS Dimuon Dataset — Download / Generate")
    print("─" * 55)

    if os.path.exists(OUTPUT):
        n = sum(1 for _ in open(OUTPUT)) - 1
        print(f"  Dataset already exists ({n:,} events). Delete to re-download.")
        sys.exit(0)

    print("  Attempting download from CERN Open Data Portal ...")
    if not try_download():
        print("  Download unavailable. Switching to synthetic generation.")
        generate_synthetic()

    print("─" * 55)
    print("  Done. Run the notebook to start the analysis.")
    print("─" * 55)
