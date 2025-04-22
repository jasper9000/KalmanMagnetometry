"""
Author: Jasper Riebesehl, 2025
"""
import jax.numpy as jnp
import numpy as np


def prepare_signal_stft(signal, f_sampling, blocklength_s, f_0_estimate, L=1):
    t = jnp.arange(signal.shape[0]) / f_sampling
    T = int(jnp.floor(blocklength_s * f_sampling))

    N = len(signal) // T # number of (full) blocks. Disregard last samples that do not fill a block
    signal_blocks = signal[:N*T].reshape(N, T, 1)
    t_blocks = t[T//2::T][:N]

    MM = f_0_estimate * T / (f_sampling)
    M = int(jnp.round(MM)) 
    delta_f_ini = MM - M
    print(f"Effective bandpass filter centered at {M/T * f_sampling:.2f} Hz with {2*(L+1)/T * f_sampling:.2f} Hz bandwidth.")

    # calculate fft per block
    signal_rfft_block = jnp.fft.rfft(signal_blocks, axis=1, norm="forward")
    # only select 2L + 1 samples around M
    signal_rfft_block = signal_rfft_block[:, M-L:M+L+1]
    # stack real and imaginary parts
    y = jnp.hstack([jnp.real(signal_rfft_block), jnp.imag(signal_rfft_block)])
    return t_blocks, y, T, N, M, delta_f_ini
