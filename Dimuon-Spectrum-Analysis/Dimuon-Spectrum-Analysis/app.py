import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import subprocess

# 1. Get the exact directory where app.py is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Add this directory to sys.path so 'src.physics_utils' can be found
sys.path.insert(0, CURRENT_DIR)
from src.physics_utils import compute_invariant_mass, fit_resonance

st.title("Dimuon Invariant Mass Analysis")
st.markdown("**CERN CMS Open Data** — Real particle collision events")

@st.cache_data
def load_data():
    # 3. Construct absolute paths looking right next to app.py
    file_path = os.path.join(CURRENT_DIR, 'data', 'dimuon.csv')
    download_script = os.path.join(CURRENT_DIR, 'data', 'download_data.py')
    
    if not os.path.exists(file_path):
        if os.path.exists(download_script):
            # 4. Run the download script, keeping the working directory here
            subprocess.run(['python', 'data/download_data.py'], check=True, cwd=CURRENT_DIR)
        else:
            st.error(f"Could not find the download script at: {download_script}. Please check your GitHub folder structure.")
            st.stop()
            
    return pd.read_csv(file_path)

# Load the data using the cached function
df = load_data()

# Process data
df['M'] = compute_invariant_mass(
    df.E1, df.px1, df.py1, df.pz1,
    df.E2, df.px2, df.py2, df.pz2
)
df = df[(df.M > 1.0) & (df.M < 130.0)]

# UI Controls
st.sidebar.header("Controls")
n_bins = st.sidebar.slider("Histogram bins", 100, 800, 400)
log_scale = st.sidebar.checkbox("Log scale", value=True)

# Plotting
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
