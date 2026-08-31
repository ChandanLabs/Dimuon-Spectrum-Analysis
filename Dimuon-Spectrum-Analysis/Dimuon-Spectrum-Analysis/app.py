import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import subprocess

sys.path.insert(0, '.')
from src.physics_utils import compute_invariant_mass, fit_resonance

st.title("Dimuon Invariant Mass Analysis")
st.markdown("**CERN CMS Open Data** — Real particle collision events")

@st.cache_data
def load_data():
    file_path = 'data/dimuon.csv'
    
    # Check if the file exists in the deployment environment
    if not os.path.exists(file_path):
        # Fallback: attempt to run the download script if it exists
        download_script = 'data/download_data.py'
        if os.path.exists(download_script):
            subprocess.run(['python', download_script], check=True)
        else:
            st.error(f"Data file '{file_path}' not found. Please upload it to your GitHub repository.")
            st.stop()
            
    return pd.read_csv(file_path)

# Load the data using the cached function
df = load_data()

df['M'] = compute_invariant_mass(
    df.E1, df.px1, df.py1, df.pz1,
    df.E2, df.px2, df.py2, df.pz2
)
df = df[(df.M > 1.0) & (df.M < 130.0)]

st.sidebar.header("Controls")
n_bins = st.sidebar.slider("Histogram bins", 100, 800, 400)
log_scale = st.sidebar.checkbox("Log scale", value=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['M'], bins=n_bins, color='steelblue', alpha=0.7)
if log_scale:
    ax.set_yscale('log')
for mv, lbl, col in [(3.097,'J/ψ','purple'),
                      (9.460,'Υ','orange'),
                      (91.19,'Z boson','blue')]:
    ax.axvline(mv, color=col, ls='--', lw=1.5, label=lbl)
ax.set_xlabel('Invariant Mass [GeV/c²]')
ax.set_ylabel('Event count')
ax.legend()
st.pyplot(fig)

st.metric("Total events", f"{len(df):,}")
st.metric("Z boson candidates", f"{((df.M>85)&(df.M<97)).sum():,}")
