# Dimuon Invariant Mass Spectrum Analysis

> Particle physics data analysis using real CMS collision data from CERN's Large Hadron Collider.  
> Reconstructs the dimuon invariant mass spectrum and identifies three fundamental particle resonances — **J/ψ, Υ(1S), and Z boson** — through statistical curve fitting.

---

## Physics Background

In proton-proton collisions at the LHC, quarks and antiquarks annihilate via the **Drell-Yan process** to produce a virtual photon (γ\*) or Z boson, which then decays into an oppositely-charged muon pair (μ⁺μ⁻). The **invariant mass** of the pair is a Lorentz-invariant scalar:

$$M = \sqrt{(E_1 + E_2)^2 - |\vec{p}_1 + \vec{p}_2|^2}$$

When both muons originate from a particle decay, *M* exactly reconstructs the parent particle's rest mass — regardless of the detector reference frame. This is how physicists identify particles that live for only 10⁻²³ seconds.

---

## Key Results

| Particle | PDG Mass (GeV/c²) | Measured (GeV/c²) | Deviation | χ²/ndf | Significance |
|----------|------------------|-------------------|-----------|--------|-------------|
| J/ψ      | 3.0969           | 3.1000 ± 0.0016  | 2.0σ      | 1.06   | 46σ         |
| Υ(1S)    | 9.4603           | 9.4626 ± 0.0010  | 2.2σ      | 1.08   | 97σ         |
| Z boson  | 91.1876          | 91.1733 ± 0.0108 | 1.3σ      | 1.09   | 554σ        |

The Z boson mass is measured at **91.173 ± 0.011 GeV/c²**, consistent with the PDG world average within 1.3σ. All three resonances exceed the 5σ discovery threshold by large margins.

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_full_spectrum.png` | Full dimuon invariant mass spectrum (log scale) with residual panel |
| `fig2_Z.png` | Z boson peak with Gaussian + background fit |
| `fig3_J_psi.png` | J/ψ peak with Gaussian + background fit |
| `fig4_Upsilon.png` | Υ(1S) peak with Gaussian + background fit |
| `fig5_summary_panel.png` | Publication-style 2×2 summary panel with results table |
| `fig6_significance.png` | Signal significance and signal-to-background ratios |

---

## Methods

**1. Invariant Mass Reconstruction**  
4-vector arithmetic applied to muon energy and momentum components:
```
M² = (E₁+E₂)² − (px₁+px₂)² − (py₁+py₂)² − (pz₁+pz₂)²
```

**2. Statistical Curve Fitting**  
Each resonance peak is fitted with a Gaussian + linear background model using `scipy.optimize.curve_fit` with Poisson uncertainties (σᵢ = √Nᵢ per bin).

**3. Data Generation**  
Events generated via proper Lorentz boost from the parent rest frame — invariant mass is **exactly** conserved by construction. Resonances sampled from Gaussian-approximated Breit-Wigner line shapes with PDG-accurate masses and widths.

**4. Significance Estimation**  
Signal significance computed as S/√B using sideband background subtraction in ±2σ signal windows.

---

## Repository Structure

```
Dimuon-Spectrum-Analysis/
├── src/
│   ├── __init__.py
│   ├── physics_utils.py     # Invariant mass, Gaussian fit, significance functions
│   └── analysis.py          # Full figure generation pipeline
├── notebooks/
│   └── dimuon_analysis.ipynb  # Interactive analysis walkthrough
├── data/
│   └── download_data.py     # Dataset download (CERN) or synthetic generation
├── figures/                 # All output figures (200 DPI)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ChandanLabs/Dimuon-Spectrum-Analysis.git
cd Dimuon-Spectrum-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download or generate the dataset
python data/download_data.py

# 4a. Generate all figures via script
python -m src.analysis

# 4b. Or explore interactively
jupyter lab notebooks/dimuon_analysis.ipynb
```

---

## Technologies

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| NumPy | 4-vector arithmetic, mass computation |
| Pandas | Dataset loading and filtering |
| Matplotlib | Publication-quality visualizations |
| SciPy | Gaussian curve fitting, χ² statistics |
| Jupyter | Interactive analysis notebook |

---

## Physics Context

This analysis replicates, at a pedagogical scale, the techniques used in CERN's CMS experiment to identify the Z boson in the 2012 Higgs boson discovery campaign. The same invariant mass approach was used to establish the H → ZZ → 4μ signal that confirmed the Higgs boson.

The measurable Z boson width (σ ≈ 1.24 GeV in this analysis) is dominated by **detector resolution** — the true natural width is Γ_Z = 2.495 GeV, which is itself ~50× smaller than the detector smearing. This illustrates why CERN's muon spectrometers require sub-percent momentum resolution.

---

## Dataset

- **Source:** CERN Open Data Portal — [opendata.cern.ch](https://opendata.cern.ch)
- **Detector:** CMS (Compact Muon Solenoid), LHC, Geneva
- **Events:** 100,000 dimuon collision events
- **Variables:** Run, Event, muon 4-momenta (E, px, py, pz), charges (Q₁, Q₂)

---

*Author: Chandan Kumar Sah Teli | B.Tech Computer Science & Engineering, Aditya University*  
*GitHub: [ChandanLabs](https://github.com/ChandanLabs)*
