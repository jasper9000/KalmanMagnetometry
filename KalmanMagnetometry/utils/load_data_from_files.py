"""
Author: Jasper Riebesehl, 2025
"""
import pandas as pd
import jax.numpy as jnp

def load_csv_file(filename, to_pT=True):
    data = pd.read_csv(str(filename), delimiter="\t", header=None, skiprows=1)
    signal = jnp.array(data[0])
    L_signal = len(signal)
    if to_pT:
        signal = signal*1e12
    return signal, L_signal


