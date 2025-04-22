"""
Author: Jasper Riebesehl, 2025
"""
import numpy as np

def gen_static_signal_no_noise(L, amp, freq, meas_noise_var, f_sampling=500):
    t = np.arange(L) / f_sampling
    mn = np.sqrt(meas_noise_var) * np.random.randn(L)
    y = amp * np.sin(2*np.pi*freq*t) + mn
    return y, mn

def gen_static_signal_white_freq_noise(L, amp, freq, meas_noise_var, freq_noise_var, f_sampling=500):
    t = np.arange(L) / f_sampling
    mn = np.sqrt(meas_noise_var) * np.random.randn(L)
    fn = np.sqrt(freq_noise_var) * np.random.randn(L)
    y = amp * np.sin(2*np.pi*freq*t) + mn
    return y, mn, fn

def gen_amp_decay_signal_no_noise(L, amp_0, t2_time, freq, meas_noise_var, f_sampling=500):
    t = np.arange(L) / f_sampling
    mn = np.sqrt(meas_noise_var) * np.random.randn(L)
    amps = amp_0 * np.exp(t/t2_time)
    y = amps * np.sin(2*np.pi*freq*t) + mn
    return y, amps, mn

def gen_chirped_sin(f_array, f_sampling):
    L = len(f_array)
    # t = np.linspace(0, L) / f_sampling
    phase = 2 * np.pi * np.cumsum(f_array / f_sampling)
    return np.sin(phase)

def generate_sine_wave_with_fn(t, frequency_func, fn):
    dt = t[1] - t[0]  # Time step
    frequency = frequency_func(t)
    pn = np.cumsum(fn) * (2*np.pi*dt)
    # fn = np.cumsum(pn) * dt

    # Integrate the frequency to get phase
    phase = 2 * np.pi * np.cumsum(frequency) * dt
    phase += pn

    # Generate the sine wave
    y = np.sin(phase)

    # Frequency with noise contribution
    frequency_with_noise = frequency + fn
    return y, frequency_with_noise, phase
