"""JAX implementation of the extended Kalman smoother.

Author: Jasper Riebesehl, 2025
"""

# imports
import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

from .state_space_model import JaxStateModel

@jit
def run_eks_learnable(
    state_model_jax: JaxStateModel,
    m_filtered,
    P_filtered,
    params):

    L = m_filtered.shape[0]
    DIM_Q = m_filtered.shape[1]
    f, df, _, _, _, _ = state_model_jax.get_functions()

    if "GQ" in params:
        GQ = params["GQ"]
    else:
        GQ = jnp.eye(DIM_Q)

    # init arrays
    m_smoothed = jnp.zeros_like(m_filtered)
    P_smoothed = jnp.zeros_like(P_filtered)
    Gs = jnp.zeros_like(P_filtered)

    # set init values at k=L-1
    m_filt_ini = m_filtered.at[-1, :, :].get()
    P_filt_ini = P_filtered.at[-1, :, :].get()

    m_smoothed = m_smoothed.at[-1,:,:].set(m_filt_ini)
    P_smoothed = P_smoothed.at[-1,:,:].set(P_filt_ini)

    @jit
    def eks_iter_jax(i, loop_iterables):
        k = L - i
        m_smoothed, P_smoothed, Gs, m_smooth_last, P_smooth_last, m_filtered, P_filtered, params = loop_iterables

        f_k = f(m_filtered.at[k, :, :].get(), k, params)
        df_k = df(m_filtered.at[k, :, :].get(), k, params)
        P_pred = df_k @ P_filtered.at[k, :, :].get() @ df_k.T + GQ@params["Q"]@GQ.T

        G = jnp.linalg.solve(P_pred.T, df_k @ P_filtered[k].T).T
        m = m_filtered.at[k, :, :].get() + G @ (m_smooth_last - f_k)

        # use stabilized recursion
        I = jnp.eye(DIM_Q)
        P = G@(P_smooth_last + GQ@params["Q"]@GQ.T)@G.T + (I - G@df_k)@P_filtered.at[k, :, :].get()@(I - G@df_k).T

        m_smooth_last = m
        P_smooth_last = P

        m_smoothed = m_smoothed.at[k, :, :].set(m)
        P_smoothed = P_smoothed.at[k, :, :].set(P)
        Gs = Gs.at[k, :, :].set(G)

        return m_smoothed, P_smoothed, Gs, m_smooth_last, P_smooth_last, m_filtered, P_filtered, params

    loop_iterables = (m_smoothed, P_smoothed, Gs, m_filt_ini, P_filt_ini, m_filtered, P_filtered, params)
    res = fori_loop(2, L+1, eks_iter_jax, loop_iterables)
    return res[:3]


@jit
def run_eks_learnable_EM(
    state_model_jax: JaxStateModel,
    m_filtered,
    P_filtered,
    params,
    y,
    k0=0,
    A_old=None):
    if k0 != 0:
        print("WARNING, k0 != 0 not implemented!")

    L = m_filtered.shape[0]
    DIM_R = y.shape[1]
    DIM_Q = m_filtered.shape[1]

    if "GQ" in params:
        GQ = params["GQ"]
    else:
        GQ = jnp.eye(DIM_Q)
    
    f, df, _, h, dh, _ = state_model_jax.get_functions()

    # set init values at k=L-1
    m_smoothed_last = m_filtered.at[-1, :, :].get()
    P_smoothed_last = P_filtered.at[-1, :, :].get()

    # for EM calculation of Q
    C = jnp.zeros_like(P_smoothed_last)
    PHI = jnp.zeros_like(P_smoothed_last)
    SIGMA = jnp.zeros_like(P_smoothed_last)
    OMEGA = jnp.zeros((DIM_R, DIM_R))

    @jit
    def eks_iter_jax(i, loop_iterables):
        k = L - i
        m_smooth_last, P_smooth_last, C, PHI, SIGMA, OMEGA = loop_iterables

        f_k = f(m_filtered.at[k, :, :].get(), k+k0, params)
        df_k = df(m_filtered.at[k, :, :].get(), k+k0, params)
        P_pred = df_k @ P_filtered.at[k, :, :].get() @ df_k.T + GQ@params["Q"]@GQ.T

        G = P_filtered.at[k, :, :].get() @ df_k.T @ jnp.linalg.inv(P_pred)
        m = m_filtered.at[k, :, :].get() + G @ (m_smooth_last - f_k)
        P = P_filtered.at[k, :, :].get() + G @ (P_smooth_last - P_pred) @ G.T


        C += P_smooth_last@G.T + m_smooth_last@m.T
        PHI += P + m @ m.T
        SIGMA += P_smooth_last + m_smooth_last@m_smooth_last.T
        y_k = y.at[k, :, :].get()

        h_k = h(m, k, params)
        dH = dh(m, k, params)
        yHx = y_k - h_k
        OMEGA += yHx@yHx.T - dH@P@dH.T

        m_smooth_last = m
        P_smooth_last = P

        return m_smooth_last, P_smooth_last, C, PHI, SIGMA, OMEGA

    loop_iterables = (m_smoothed_last, P_smoothed_last, C, PHI, SIGMA, OMEGA)
    res = fori_loop(2, L+1, eks_iter_jax, loop_iterables)

    m_smoothed_last, P_smoothed_last, C, PHI, SIGMA, OMEGA = res

    # EM calculations
    C /= (L-1)
    SIGMA /= (L-1)
    PHI /= (L-1)
    OMEGA /= (L-1)

    # A = C@jnp.linalg.inv(PHI)
    # instead of learned A, use F because we want that to be constant
    A = df(m_smoothed_last, 0, params)

    if A_old is not None:
        Q_new = SIGMA - C@A_old.T - A_old@C.T + A_old@PHI@A_old.T
    else:
        Q_new = SIGMA - C@A.T - A@C.T + A@PHI@A.T
    R_new = OMEGA

    return m_smoothed_last, P_smoothed_last, Q_new, R_new, A
