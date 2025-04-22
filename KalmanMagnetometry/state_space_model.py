"""
State model for the Kalman filter for the JAX implementation.

Author: Jasper Riebesehl, 2025
"""
from copy import deepcopy
from inspect import getsource
from jax import jit, jacfwd, hessian
from jax.tree_util import register_pytree_node_class, Partial
from jaxlib.xla_extension import PjitFunction
from jax.lax import cond
import jax.numpy as jnp


@register_pytree_node_class # make state model class a pytree node to avoid jit errors
class JaxStateModel():
    """State model for the Kalman filter.
    Args:
        f (function): The state transition function.
        df (function): The Jacobian of the state transition function.
        h (function): The measurement function.
        dh (function): The Jacobian of the measurement function.
    """

    def __init__(self, f, h, df=None, dh=None, dff=None, dhh=None, is_compiled=jnp.array(False), f_orig=None, h_orig=None) -> None:
        self.save_original_functions(f, f_orig, h, h_orig)

        self.f = f
        self.h = h

        # optionals
        self.df  = df
        self.dff = dff
        self.dh  = dh
        self.dhh = dhh
        self.is_compiled = is_compiled

    def compile(self):
        """Compile the state model functions."""
        if type(self.f) is not Partial:
            self.f = jit(self.f)
        if self.df is None:
            self.df = jit(lambda x,k,params: jacfwd(self.f, argnums=(0,))(x,k, params)[0][:,0,:,0])
        elif type(self.df) is not Partial:
            self.df = jit(self.df)

        if self.dff is None:
            self.dff = jit(lambda x,k,params: hessian(self.f)(x,k,params)[:,0,:,0,:,0])
        elif type(self.dff) is not Partial:
            self.dff = jit(self.dff)

        if type(self.h) is not Partial:
            self.h = jit(self.h)
        if self.dh is None:
            self.dh = jit(lambda x,k,params: jacfwd(self.h, argnums=(0,))(x,k,params)[0][:,0,:,0])
        elif type(self.dh) is not Partial:
            self.dh = jit(self.dh)
        if self.dhh is None:
            self.dhh = jit(lambda x,k,params: hessian(self.h)(x,k,params)[:,0,:,0,:,0])
        elif type(self.dhh) is not Partial:
            self.dhh = jit(self.dhh)

        self.functions_to_pytree()
        self.is_compiled = jnp.array(True)
    
    def functions_to_pytree(self):
        self.f = Partial(self.f)
        self.df = Partial(self.df)
        self.dff = Partial(self.dff)
        self.h = Partial(self.h)
        self.dh = Partial(self.dh)
        self.dhh = Partial(self.dhh)

    def get_functions(self):
        """Get the state model functions."""
        return self.f, self.df, self.dff, self.h, self.dh, self.dhh
    
    def save_original_functions(self, f, f_orig, h, h_orig):
        if f_orig is not None:
            self.f_orig = f_orig
        else:
            self.f_orig = deepcopy(f)
        if h_orig is not None:
            self.h_orig = h_orig
        else:
            self.h_orig = deepcopy(h)

    def get_original_functions(self):
        return self.f_orig, self.h_orig
    
    def get_functions_sourcecode(self):
        return [getsource(g) for g in self.get_original_functions()]

    def tree_flatten(self):
        children = (self.f, self.h, self.df, self.dh, self.dff, self.dhh, self.is_compiled)  # arrays / dynamic values
        aux_data = (self.f_orig, self.h_orig)  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

