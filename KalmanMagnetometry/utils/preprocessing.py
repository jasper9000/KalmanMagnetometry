"""
Author: Jasper Riebesehl, 2025
"""
from scipy.signal import butter, filtfilt, sosfiltfilt, find_peaks
import numpy as np

def highpass_filter(signal, cutoff, f_sampling):
    nyquist = 0.5 * f_sampling
    normal_cutoff = cutoff / nyquist
    sos = butter(10, normal_cutoff, btype='high', analog=False, output='sos')
    return sosfiltfilt(sos, signal)

def notch_filter(signal, f_stop_band, f_sampling):
    nyquist = 0.5 * f_sampling
    low = f_stop_band[0]
    high = f_stop_band[1]
    low = low / nyquist
    high = high / nyquist
    sos = butter(10, [low, high], btype='bandstop', analog=False, output='sos')
    return sosfiltfilt(sos, signal)
