"""
Author: Jasper Riebesehl, 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def plot_tracked_vars_stft(data_filter, n_sigma=2):
    fig, ax = plt.subplots(2,1, figsize=(10,6))
    
    ### frequency
    # ekf
    ax[0].plot(data_filter["t_blocks"], data_filter["f_ekf"], label=f"EKF Frequency +- {n_sigma} sigma")
    ax[0].fill_between(
        data_filter["t_blocks"],
        data_filter["f_ekf"] - n_sigma*data_filter["sigma_f_ekf"],
        data_filter["f_ekf"] + n_sigma*data_filter["sigma_f_ekf"],
        alpha=0.5)
    # eks
    ax[0].plot(data_filter["t_blocks"], data_filter["f_eks"], label=f"EKS Frequency +- {n_sigma} sigma")
    ax[0].fill_between(
        data_filter["t_blocks"],
        data_filter["f_eks"] - n_sigma*data_filter["sigma_f_eks"],
        data_filter["f_eks"] + n_sigma*data_filter["sigma_f_eks"],
        alpha=0.5)

    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Frequency [Hz]")

    ## amps
    # ekf
    ax[1].plot(data_filter["t_blocks"], data_filter["amp_ekf"], label=f"EKF Amplitude +- {n_sigma} sigma")
    ax[1].fill_between(
        data_filter["t_blocks"],
        data_filter["amp_ekf"] - n_sigma*data_filter["sigma_amp_ekf"],
        data_filter["amp_ekf"] + n_sigma*data_filter["sigma_amp_ekf"],
        alpha=0.5)
    # eks
    ax[1].plot(data_filter["t_blocks"], data_filter["amp_eks"], label=f"EKS Amplitude +- {n_sigma} sigma")
    ax[1].fill_between(
        data_filter["t_blocks"],
        data_filter["amp_eks"] - n_sigma*data_filter["sigma_amp_eks"],
        data_filter["amp_eks"] + n_sigma*data_filter["sigma_amp_eks"],
        alpha=0.5)

    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Amplitude [pT]")

    for a in ax:
        a.grid()
        a.legend(loc="upper right", fontsize=9)
        # no offset
        a.get_yaxis().get_major_formatter().set_useOffset(False)
    fig.tight_layout()
    return fig, ax



def plot_optimization_progress_stft(progress, M,T,f_sampling):
    fig, ax = plt.subplots(3,3, figsize=(10,6))

    DIM_Q = progress["Q"][-1].shape[0]
    DIM_R = progress["R"][-1].shape[0]
    # R
    try:
        for j in range(DIM_R):
            ax[0,0].semilogy(progress["R"][:,j,j])
        ax[0,0].set_title("R")
    except:
        pass
    # Q
    for i in range(DIM_Q):
    #     if (gq is not None) and gq[i,i]!=0.0:
        ax[0,1].semilogy(progress["Q"][:,i,i].squeeze(), color=f"C{i}")
    # ax[2,1].semilogy(progress["Q"].squeeze())
    ax[0,1].set_title("Q")

    # 0,2 loss
    ax[0,2].plot(np.array(progress.losses)[10:], color="black")
    ax[0,2].set_title("loss")

    # #1,0 P0
    try:
        for i in range(DIM_Q):
            ax[1,0].semilogy(progress["P_ini"][:,i,i].squeeze(), label=f"P_ini_{i}")
        ax[1,0].legend()
    except:
        pass

    # # 2,0 m0
    # amp_ini
    ax[2,0].plot(4*np.pi*progress["m_ini"][:,0].squeeze()**2, label=f"A_0")
    ax[2,0].set_title("A_0")

    # f_ini
    ax[2,1].plot((progress["m_ini"][:,2].squeeze()+M)/ (T) *f_sampling, label=f"f_0")
    ax[2,1].set_title("f_0")

    for a in ax.flatten():
        a.grid()
        # a.set_xscale("log")

    fig.tight_layout()
    return fig, ax