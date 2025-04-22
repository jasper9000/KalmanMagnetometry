"""
Author: Jasper Riebesehl, 2025
"""
import jax
import jax.numpy as jnp
import numpy as np
import noise_characterization as nc
import copy


def get_loss_fun_EM(state_model):
    @jax.jit
    def loss_EM(params, y):
        m_est, P_est, ll, _ = nc.jax.run_ekf_learnable(state_model, y, params, k0=0)
        return (ll/ y.shape[0], (m_est, P_est))
    return loss_EM


def training_loop_EM(progress, y_train, state_model, params, max_iter, verbose=True, max_iters=10, alpha_Q=0.0, alpha=0.8):
    L_train = y_train.shape[0]
    progress.L_train = L_train

    loss_EM = get_loss_fun_EM(state_model)

    err = 1e4
    i = 0

    DIM_Q = params["Q"].shape[0]
    A_new = jnp.eye(DIM_Q)
    P_new = params["P_ini"]
    # while (i == 0) | ((i < max_iter) & (err >= tol)):
    while 1:
        vals = loss_EM(params, y_train)

        ll = vals[0]
        m_est = vals[1][0]
        P_est = vals[1][1]

        if np.isnan(ll):
            print(ll)
            break

        m_smooth_first, P_smooth_first, Q_new, R_new, A_new = nc.jax.run_eks_learnable_EM(state_model, m_est, P_est, params, y_train, A_old=A_new)

        R_new = jnp.real(jax.scipy.linalg.sqrtm((R_new@R_new.T)))
        params["R"] = alpha * params["R"] + (1-alpha)* R_new

        Q_new = jnp.real(jax.scipy.linalg.sqrtm((Q_new@Q_new.T)))
        params["Q"] = alpha_Q * params["Q"] + (1-alpha_Q)*Q_new

        PPP = P_smooth_first + (m_smooth_first - params["m_ini"]) @ (m_smooth_first - params["m_ini"]).T
        P_new = alpha * P_new + (1-alpha)*PPP

        params["P_ini"] = P_new
        params["m_ini"] = alpha * params["m_ini"] + (1 - alpha) * m_smooth_first

        # logging
        save_params = copy.deepcopy(params)
        progress.add(ll, {k:v for k,v in save_params.items() if k in save_params.keys()})

        i += 1
        if verbose:
            print(f"L_TRAIN: {L_train}, I: {i}/{max_iter}, LL: {ll:.4f}, GRADS: {err:.3e}")

        if (i>= max_iter):
            break

    return params, progress


def perform_EM(signal, l_train_list, max_iter_list, state_model, params, progress=None, **kwargs):
    if progress is None:
        progress = nc.jax.RunProgress(0)

    y_train_list = [signal[:l] for l in l_train_list]
    for j, y_train in enumerate(y_train_list):
        params, progress = training_loop_EM(progress, y_train, state_model, params, max_iter_list[j], **kwargs)

    return params, progress


def EM_bisection(signal, state_model, params_ini, l_train_em, n_iter_burn = 30, n_iter_bisect = 5, n_max_switches = 5, q_factor_ini=10.0, idx_qs=list(), alpha_Q=0.0, bisection_root_factor=0.5, max_iters=100, verbose=True, **kwargs):
    # get initial params
    params = copy.deepcopy(params_ini)
    params_last = copy.deepcopy(params)

    DIM_Q = params["Q"].shape[-1]

    n_qs = len(idx_qs)
    qs_ini = np.diag(params["Q"])[idx_qs]

    params, progress = perform_EM(signal, [l_train_em], [n_iter_burn], state_model, params, progress=None, verbose=False, **kwargs)

    l_train_list = [l_train_em]
    max_iter_list = [n_iter_bisect]

    qs_last_sign = np.zeros(n_qs, dtype=int)
    qs_factor = np.array([q_factor_ini] * n_qs)
    q_factors_abort = qs_factor**(1/(2**n_max_switches))


    iter = 0
    while (np.any(qs_factor > q_factors_abort)) and (iter < max_iters):
        if verbose:
            print(f"BISECTION ITER: {iter}")

        # perform few EM steps
        params, progress = perform_EM(signal, l_train_list, max_iter_list, state_model, params, progress=progress, verbose=False, alpha_Q=alpha_Q, **kwargs)

        # get final values
        qs_end = np.diag(params["Q"])[idx_qs]


        for j, idx in enumerate(idx_qs):
            qs_ini[j], qs_factor[j], qs_last_sign[j] = bisection_esque_algo(params_last["Q"][idx,idx], qs_end[j], qs_factor[j], qs_last_sign[j], bisection_root_factor=bisection_root_factor)

        iter += 1

        gg = jnp.eye(DIM_Q)
        for j, idx in enumerate(idx_qs):
            gg = gg.at[idx,idx].set((qs_ini[j] / params_last["Q"].at[idx,idx].get())**0.5)

        params["Q"] = gg@params["Q"]@gg.T
        params_last = copy.deepcopy(params)


    return params, progress


def bisection_esque_algo(val_start, val_end, val_factor, val_last_sign, bisection_root_factor=0.5):
    val_diff = val_end - val_start
    if (val_diff > 0):
        if (val_last_sign >= 0):
            new_val = val_start * val_factor
        else:
            val_factor = val_factor**bisection_root_factor
            new_val = val_start * val_factor
        val_last_sign = 1
    else:
        if (val_last_sign <= 0):
            new_val = val_start / val_factor
        else:
            val_factor = val_factor**bisection_root_factor
            new_val = val_start / val_factor
        val_last_sign = -1
    return new_val, val_factor, val_last_sign