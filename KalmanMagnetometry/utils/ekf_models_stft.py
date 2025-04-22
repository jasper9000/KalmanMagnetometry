"""
Author: Jasper Riebesehl, 2025
"""
import jax.numpy as jnp
from ..state_space_model import JaxStateModel

def get_stft_model_int_freq_int_amp(M, L, T):
    DIM_Q = 5
    DIM_R = 2 * (2*L+1)

    A = jnp.array(
        [
            [1, 0, 0, 0, 1e-2], # amp
            [0, 1, 2*jnp.pi, 0, 0], # phase
            [0, 0, 1, 1e-4, 0], # freq
            [0, 0, 0, 1, 0], # integrated frequency
            [0, 0, 0, 0, 1]# integrated amp
        ]
        )

    ### DEFINE STATE MODEL ###
    def f_jax(x,k,params):
        return A@x

    def h_jax(x,k,params):
        m = jnp.arange(M-L, M+L+1)

        nominator1 = jnp.exp(1j * x[1]) * (1 - jnp.exp(2j*jnp.pi*x[2]))
        nominator2 = jnp.exp(-1j * x[1]) * (1 - jnp.exp(-2j*jnp.pi*x[2]))
        denominator1 = (1 - jnp.exp( 1j*(M - m + x[2]) / T ))
        denominator2 = (1 - jnp.exp( 1j*(-M - m - x[2]) / T ))

        f_y = (x[0])**2 / (T) * (nominator1 / denominator1 + nominator2 / denominator2)

        r_y = jnp.real(f_y)
        i_y = jnp.imag(f_y)
        y = jnp.vstack([r_y, i_y])
        return y.reshape(-1,1)

    GQ = jnp.diag(jnp.array([0.0,0.0,0.0, 1.0, 1.0]))

    sm = JaxStateModel(f_jax, h_jax)
    sm.compile()
    return sm, DIM_Q, DIM_R, GQ

