"""
Author: Jasper Riebesehl, 2025
"""
import numpy as np
import jax

class RunProgress():
    def __init__(self, L_train, L_test=-1):
        self.losses = []
        self.norm_losses = []
        self.losses_test = []
        self.norm_losses_test = []
        self.params = []
        self.iterations = []
        self.L_train = L_train
        self.L_test = L_test

    def add(self, loss, params, loss_test=None, iteration=None):
        if iteration is None:
            iteration = len(self.losses)
        if loss_test is None:
            loss_test = -1
        
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.norm_losses.append(loss/self.L_train)
        self.losses_test.append(loss_test)
        self.norm_losses_test.append(loss_test/self.L_test)
        self.params.append(params)

    def get_losses(self):
        return np.array(self.losses)
    
    def get_norm_losses(self):
        return np.array(self.norm_losses)
    
    def get_iters(self):
        return np.array(self.iterations)
    
    def get_param_keys(self):
        if len(self.params) > 0:
            return self.params[0].keys()
    
    def get_params(self):
        if len(self.params) == 0:
            return None
        params = {}
        for key in self.params[0].keys():
            params[key] = []
            for param_dict in self.params:
                params[key].append(param_dict[key])
            params[key] = np.array(params[key])
        return params
    
    get_last_params = lambda self: self.params[-1]

    def __getitem__(self, key):
        if len(self.params) == 0:
            raise IndexError("No parameters have been recorded yet.")
        if key in self.params[0].keys():
            return np.array([d[key] for d in self.params])
        elif key.lower() in ["loss", "losses", "ll"]:
            return self.get_losses()
        elif key.lower() in ["i", "iter", "iters", "iterations"]:
            return self.get_iters()
        else:
            raise KeyError(f"Key {key} not found in recorded parameters.")
        
    def save_to_file(self, filename):
        np.savez(filename, losses=self.get_losses(), iterations=self.get_iters(), params=self.get_params())

    @staticmethod
    def load_from_file(filename):
        data = np.load(filename, allow_pickle=True)
        progress = RunProgress()
        progress.losses = data["losses"]
        progress.iterations = data["iterations"]
        progress.params = data["params"]
        return progress
    
    @staticmethod
    def determine_key_dim(val):
        if type(val) is int or type(val) is float:
            return 1
        elif isinstance(val, jax.numpy.ndarray):
            return val.ndim
        else:
            raise ValueError(f"Type {type(val)} not supported.")
    
    @staticmethod
    def determine_n_cols(val):
        if isinstance(val, jax.numpy.ndarray) and val.ndim >= 2:
            return val.shape[1]
        else:
            return 0
        