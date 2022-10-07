import importlib
from functools import partial

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from jax import grad, jit, random, vmap
from jax_md import smap

from . import lnn, models


def colnum(i, j, N):
    """Gives linear index for upper triangle matrix.
    """
    assert (j >= i), "j >= i, Upper Triangle indices."
    assert (i < N) and (j < N), "i<N & j<N where i and \
            j are atom type and N is number of species."
    return int(i*N - i*(i-1)/2 + j-i + 1)


def pair2mat(fn, displacement_or_metric, species, parameters,
             ignore_unused_parameters=True,
             reduce_axis=None,
             keepdims=False,
             use_onehot=False,
             **kwargs):
    kwargs, param_combinators = smap._split_params_and_combinators(kwargs)

    merge_dicts = partial(jax_md.util.merge_dicts,
                          ignore_unused_parameters=ignore_unused_parameters)
    d = lnn.t1(displacement=displacement_or_metric)
    if species is None:
        def fn_mapped(R: smap.Array, **dynamic_kwargs) -> smap.Array:
            _kwargs = merge_dicts(kwargs, dynamic_kwargs)
            _kwargs = smap._kwargs_to_parameters(
                None, _kwargs, param_combinators)
            dr = d(R)
            # NOTE(schsam): Currently we place a diagonal mask no matter what function
            # we are mapping. Should this be an option?
            return smap.high_precision_sum(fn(dr, **_kwargs),
                                           axis=reduce_axis, keepdims=keepdims) * smap.f32(0.5)

    elif jax_md.util.is_array(species):
        species = np.array(species)
        smap._check_species_dtype(species)
        species_count = int(np.max(species) + 1)
        if reduce_axis is not None or keepdims:
            raise ValueError

        def onehot(i, j, N):
            col = colnum(i, j, species_count)
            oneh = jnp.zeros(
                (N, colnum(species_count-1, species_count-1, species_count)))
            oneh = jax.ops.index_update(oneh, jnp.index_exp[:, int(col-1)], 1)
            return oneh

        def pot_pair_wise():
            if use_onehot:
                def func(i, j, dr, **s_kwargs):
                    dr = jnp.linalg.norm(dr, axis=1, keepdims=True)
                    ONEHOT = onehot(i, j, len(dr))
                    h = vmap(models.forward_pass, in_axes=(
                        None, 0))(parameters["ONEHOT"], ONEHOT)
                    dr = jnp.concatenate([h, dr], axis=1)
                    return smap.high_precision_sum(
                        fn(dr, params=parameters["PEF"], **s_kwargs))
                return func
            else:
                def func(i, j, dr, **s_kwargs):
                    return smap.high_precision_sum(
                        fn(dr, **parameters[i][j-i], **s_kwargs))
                return func

        pot_pair_wise_fn = pot_pair_wise()

        def fn_mapped(R, **dynamic_kwargs):
            U = smap.f32(0.0)
            for i in range(species_count):
                for j in range(i, species_count):
                    _kwargs = merge_dicts(kwargs, dynamic_kwargs)
                    s_kwargs = smap._kwargs_to_parameters(
                        (i, j), _kwargs, param_combinators)
                    Ra = R[species == i]
                    Rb = R[species == j]
                    if j == i:
                        dr = d(Ra)
                        dU = pot_pair_wise_fn(i, j, dr, **s_kwargs)
                        U = U + smap.f32(0.5) * dU
                    else:
                        dr = vmap(vmap(displacement_or_metric, in_axes=(0, None)), in_axes=(
                            None, 0))(Ra, Rb).reshape(-1, Ra.shape[1])
                        dU = pot_pair_wise_fn(i, j, dr, **s_kwargs)
                        U = U + dU
            return U
    return fn_mapped


def map_parameters(fn, displacement, species, parameters, **kwargs):
    mapped_fn = lnn.MAP(fn)

    def f(x, *args, **kwargs):
        out = mapped_fn(x, *args, **kwargs)
        return out
    return pair2mat(f, displacement, species, parameters, **kwargs)


class VV_unroll():
    def __init__(self, R, dt=1):
        self.R = R
        self.dt = dt

    def get_position(self):
        r = self.R[1:-1]
        return r

    def get_acceleration(self, dt=None):
        r = self.R[1:-1]
        r_minus = self.R[:-2]
        r_plus = self.R[2:]
        if dt is not None:
            return (r_plus + r_minus - 2*r)/dt**2
        else:
            return (r_plus + r_minus - 2*r)/self.dt**2

    def get_velocity(self, dt=None):
        r_minus = self.R[:-2]
        r_plus = self.R[2:]
        if dt is not None:
            return (r_plus - r_minus)/2/dt
        else:
            return (r_plus - r_minus)/2/self.dt

    def get_kin(self, dt=None):
        return self.get_position(), self.get_velocity(dt=dt), self.get_acceleration(dt=dt)


class States:
    def __init__(self, state=None, const_size=True):
        if state is None:
            self.isarrays = False
            self.const_size = const_size
            self.position = []
            self.velocity = []
            self.force = []
            if self.const_size:
                self.mass = None
            else:
                self.mass = []
        else:
            self.position = [state.position]
            self.velocity = [state.velocity]
            self.force = [state.force]
            if self.const_size:
                self.mass = state.mass
            else:
                self.mass = [state.mass]

    def add(self, state):
        self.position += [state.position]
        self.velocity += [state.velocity]
        self.force += [state.force]
        if self.const_size:
            if self.mass is None:
                self.mass = state.mass
        else:
            self.mass += [state.mass]

    def fromlist(self, states, const_size=True):
        out = States(const_size=const_size)
        for state in states:
            out.add(state)
        return out

    def makearrays(self):
        if not(self.isarrays):
            self.position = jnp.array(self.position)
            self.velocity = jnp.array(self.velocity)
            self.force = jnp.array(self.force)
            self.mass = jnp.array([self.mass])
            self.isarrays = True

    def get_array(self):
        self.makearrays()
        return self.position, self.velocity, self.force

    def get_mass(self):
        self.makearrays()
        return self.mass

    def get_kin(self):
        self.makearrays()
        if self.const_size:
            acceleration = self.force/self.mass.reshape(1, self.mass.shape)
        else:
            acceleration = self.force/self.mass
        return self.position, self.velocity, acceleration


def reload(list_of_modules):
    for module in list_of_modules:
        try:
            print("Reload: ", module.__name__)
            importlib.reload(module)
        except:
            print("Reimports failed.")


def timeit(stmt, setup="", number=5):
    from timeit import timeit
    return timeit(stmt=stmt, setup=setup, number=number)


def factorial(n):
    if n == 0:
        return 1
    else:
        return n*factorial(n-1)


def nCk(n, k):
    return factorial(n)//factorial(k)//factorial(n-k)
