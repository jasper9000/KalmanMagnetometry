"""
Author: Jasper Riebesehl, 2025
"""
import numpy as np


def gen_dense_series_from_blocks(t_dense, t_block, y_block):
    t_sc_bounds = [ii * np.diff(t_block)[0] for ii in range(len(t_block)+1)]
    t_digi_idx = np.digitize(t_dense, t_sc_bounds, right=False) - 1
    t_digi_idx[t_digi_idx>=t_block.shape[0]] -= 1 # this line handles samples beyond the last block
    y_dense = y_block[t_digi_idx]
    return y_dense

def gen_block_from_dense(blocksize_samples, y_dense, f_sampling):
    ## WARNING: This function currently disregards samples from the last (possibly incomplete) block [I think at least, too lazy to check]
    y_dense = y_dense.squeeze()
    if len(y_dense.shape) > 1:
        raise ValueError("y_dense shoud be 1d")
    L = y_dense.shape[0]
    n_blocks = int(np.ceil(L / blocksize_samples))
    t_block = (np.arange(0, n_blocks-1)*blocksize_samples + blocksize_samples/2) / f_sampling
    # Compute the total number of elements after padding
    total_elements = int(np.ceil(y_dense.size / blocksize_samples) * blocksize_samples)
    # Zero-pad the flattened array to the total elements
    y_dense_padded = np.pad(y_dense, (0, total_elements - y_dense.size))

    # Reshape into (bz, -1)
    y_reshaped = y_dense_padded.reshape(-1,blocksize_samples)
    if total_elements > y_dense.size:
        y_reshaped = y_reshaped[:-1,:]

    # average blockwise
    y_block = np.mean(y_reshaped, axis=1)
    y_block_std = np.std(y_reshaped, axis=1)
    return t_block, y_block, y_block_std
