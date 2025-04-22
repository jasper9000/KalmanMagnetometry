"""JAX implementation of the Extended Kalman Filter (EKF) for state estimation.

Author: Jasper Riebesehl, 2025
"""

import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

from .state_space_model import JaxStateModel
from .dimension_checks import *


@jit
def run_ekf_learnable(
    state_model_jax: JaxStateModel,
    signal,
    params,
    k0):

    check_array_dimensions(signal, params)

    # unpack params
    m_ini = params['m_ini']
    P_ini = params['P_ini']
    Q = params['Q']
    R = params['R']

    # make dimensiones easy to use
    L = signal.shape[0]
    DIM_Q = Q.shape[0]
    DIM_R = R.shape[0]

    if "GQ" in params:
        GQ = params["GQ"]
    else:
        GQ = jnp.eye(DIM_Q)
    
    # init arrays
    ms = jnp.zeros((L, DIM_Q, 1))
    Ps = jnp.zeros((L, DIM_Q, DIM_Q))
    log_likelihood = 0.0

    # init arrats
    ms = ms.at[0, :, :].set(m_ini)
    Ps = Ps.at[0, :, :].set(P_ini)
    
    f, df, _, h, dh, _ = state_model_jax.get_functions()
    i = 0
    loop_iterables = (
        ms, Ps,
        log_likelihood,
        m_ini, P_ini,
        signal,
        Q, R,
        params,
        i, k0)
    
    @jit
    def ekf_iter_jax(k, loop_iterables):
        mm, PP, log_likelihood, m_last, P_last, signal, Q, R, params, i, k0 = loop_iterables
        
        m_pred = f(m_last, k+k0-1, params)
        df_k = df(m_last, k+k0-1, params)
        P_pred = df_k @ P_last @ df_k.T + GQ@Q@GQ.T

        # # Update
        dh_k = dh(m_pred, k+k0, params)
        v_k = signal.at[i+1, :,:].get() - h(m_pred, k+k0, params)  # Innovation
        S_k = dh_k @ P_pred @ dh_k.T + R  # Innovation covariance
        S_k_pinv = jnp.linalg.inv(S_k)
        
        K = P_pred @ dh_k.T @ S_k_pinv  # Kalman gain
        m = m_pred + K @ (v_k)  # new state estimate

        # P = P_pred - K @ S_k @ K.T  # new state covariance estimate
        # use Joseph's form covariance update
        I = jnp.eye(DIM_Q)
        P = (I - K@dh_k)@P_pred@(I - K@dh_k).T + K@R@K.T

        mm = mm.at[i+1,:,:].set(m)
        PP = PP.at[i+1,:,:].set(P)

        # calculate log likelihood
        log_likelihood += 0.5 * (jnp.log(jnp.linalg.det(S_k)) + v_k.T @ S_k_pinv @ v_k)[0,0]

        m_last = m
        P_last = P
        i += 1
        # return mm, PP, log_likelihood, KK, m_last, P_last, signal, Q, R, params, i, k0
        return mm, PP, log_likelihood, m_last, P_last, signal, Q, R, params, i, k0
    
    res = fori_loop(1, L, ekf_iter_jax, loop_iterables)
    return res[:4]
