"""
Author: Jasper Riebesehl, 2025
"""
import numpy as np
import jax
from functools import partial

from ..ekf import run_ekf_learnable
from ..eks import run_eks_learnable

def apply_filter(signal, w_filter, state_model, params):
    aux_data = {}
    # check if w_filter is an int
    if isinstance(w_filter, int):
        w_filter = np.arange(w_filter)

    m_est, P_est, _,_ = run_ekf_learnable(state_model, signal[w_filter], params, w_filter[0])
    m_smooth, P_smooth, Gs = run_eks_learnable(state_model, m_est, P_est, params)
    aux_data["Gs"] = Gs
    return m_est, P_est, m_smooth, P_smooth, aux_data


def reconstruct_signal(signal, w_filter, m_est, state_model, params):
    if isinstance(w_filter, int):
        w_filter = np.arange(w_filter)

    sig_rec = jax.vmap(partial(state_model.h, params=params), in_axes=(0, 0))(m_est, w_filter.reshape(-1,1))

    residuals = signal[w_filter].squeeze() - sig_rec.squeeze()
    return sig_rec, residuals