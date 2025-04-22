"""
Author: Jasper Riebesehl, 2025
"""

def check_array_dimensions(y, params):
    L = y.shape[0]
    DIM_Q = params["Q"].shape[0]
    DIM_R = params["R"].shape[0]
    
    # assert existence of key parameters
    assert "m_ini" in params
    assert "P_ini" in params
    assert "Q" in params
    assert "R" in params
    # assert "GQ" in params

    # assert shapes of key parameters
    assert params["Q"].shape == (DIM_Q, DIM_Q)
    assert params["R"].shape == (DIM_R, DIM_R)
    assert params["m_ini"].shape == (DIM_Q, 1)
    assert params["P_ini"].shape == (DIM_Q, DIM_Q)
    assert y.shape == (L, DIM_R, 1)
