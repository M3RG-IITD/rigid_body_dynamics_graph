from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from numpy.core.fromnumeric import reshape

from .models import ReLU, SquarePlus, forward_pass


def MAP(input_fn):
    """Map vmap for first input.

    :param input_fn: function to map
    :type input_fn: function
    """
    def temp_g(x, *args, **kwargs):
        def temp_f(x):
            return input_fn(x, *args, **kwargs)
        return vmap(temp_f, in_axes=0)(x)
    return temp_g


def nonan(input_fn):
    """Apply nonan macro.

    :param input_fn: input function
    :type input_fn: function
    """
    def out_fn(*args, **kwargs):
        return jnp.nan_to_num(input_fn(*args, **kwargs))
    out_fn.__doc__ = input_fn.__doc__
    return out_fn


def describe_params(params_):
    """Print parameters.

    :param params_: Parameters
    :type params_: dict or list
    :return: description of parameters.
    :rtype: string
    """
    if isinstance(params_, dict):
        str_ = ""
        for k, params in params_.items():
            str_ = str_ + f"{k}\n" + \
                "\n".join([f"\tLayer {ind}\n\tW: {p[0].shape}, b: {p[1].shape}"
                           for ind, p in enumerate(params)])
        return str_
    else:
        return "\n".join([f"Layer {ind}\n\tW: {p[0].shape}, b: {p[1].shape}"
                          for ind, p in enumerate(params_)])


def FFLNN(x, v, params):
    x_ = x.reshape(-1,)
    return _T(v) - forward_pass(params, x_)[0]


def LNN(x, v, params):
    """
    x: Vector
    v: Vector
    """
    x_ = x.reshape(-1, )
    v_ = v.reshape(-1, )
    return forward_pass(params, jnp.vstack([x_, v_]))[0]


def _V(x, params):
    pass


def _T(v, mass=jnp.array([1.0])):
    if len(mass) != len(v):
        mass = mass[0]*jnp.ones((len(v)))
    out = mass*jnp.square(v).sum(axis=1)
    return 0.5*out.sum()


def _L(x, v, params):
    pass


def lagrangian(x, v, params):
    """
    lagrangian calls lnn._L
    x: Vector
    v: Vector
    """
    return _L(x, v, params)


def calM(x, v, params):
    return jax.hessian(lagrangian, 1)(x, v, params)


jcalM = jit(calM)


def calMinv(x, v, params):
    return jnp.linalg.pinv(calM(x, v, params))


jcalMinv = jit(calMinv)


def acceleration(x, v, params):
    Dim = x.shape[1]
    N = x.shape[0]*Dim
    M_1 = jcalMinv(x, v, params).reshape(N, N)
    dx_L = jax.grad(lagrangian, 0)(x, v, params).reshape(N, 1)
    dxdv_L = jax.jacobian(jax.jacobian(lagrangian, 1),
                          0)(x, v, params).reshape(N, N)
    out = M_1 @ (dx_L - dxdv_L @ v.reshape(N, 1))
    return out.reshape(-1, Dim)


def accelerationTV(x, v, params):
    Dim = x.shape[1]
    N = x.shape[0]
    M_1 = jnp.linalg.pinv(jax.hessian(_T, 0)(v).reshape(N*Dim, N*Dim))
    dx_L = jax.grad(lagrangian, 0)(x, v, params).reshape(-1, 1)
    out = M_1 @ (dx_L)
    return out.reshape(-1, Dim)


def accelerationFull(n, Dim, lagrangian=lagrangian, non_conservative_forces=None, external_force=None, constraints=None, jac_constraints=None):
    """ ̈q = M⁻¹(-C ̇q + Π + Υ - Aᵀ(AM⁻¹Aᵀ)⁻¹ ( AM⁻¹ (-C ̇q + Π + Υ + F ) + Adot ̇q ) + F )

    :param T: [description], defaults to _T
    :type T: [type], optional
    :param lagrangian: [description], defaults to lagrangian
    :type lagrangian: [type], optional
    """
    def inv(x, *args, **kwargs):
        return jnp.linalg.pinv(x, *args, **kwargs)

    if non_conservative_forces == None:
        def non_conservative_forces(x, v, params): return 0
    if external_force == None:
        def external_force(x, v, params): return 0
    if constraints == None:
        def constraints(x, v, params): return jnp.zeros((1, n*Dim))
    if jac_constraints==None:
        def jac_constraints(x, v, params): return jax.jacobian(constraints, 0)(x.reshape(-1), v.reshape(-1), params)

    
    eye = jnp.eye(n*Dim)

    def dL_dv(R, V, params):
        return jax.grad(lagrangian, 1)(R.reshape(n, Dim),
                                       V.reshape(n, Dim), params).flatten()

    def d2L_dv2(R, V, params):
        return jax.jacobian(dL_dv, 1)(R, V, params)
        # return eye*jnp.diag(jax.jacobian(dL_dv, 1)(R, V, params))

    def fn(x, v, params):
        N = n*Dim
        # M⁻¹ = (∂²L/∂²v)⁻¹
        M = d2L_dv2(x.flatten(), v.flatten(), params)
        M_1 = inv(M)
        # Π = ∂L/∂x
        Π = jax.grad(lagrangian, 0)(x, v, params).reshape(
            N, 1)
        # C = ∂²L/∂v∂x
        C = jax.jacobian(jax.jacobian(lagrangian, 1),
                         0)(x, v, params).reshape(N, N)
        Υ = non_conservative_forces(x, v, params)
        F = external_force(x, v, params)
        A = constraints(x.reshape(-1), v.reshape(-1), params)
        Aᵀ = A.T
        AM_1 = A @ M_1
        v = v.reshape(N, 1)
        # Ax = jax.jacobian(constraints, 0)(x.reshape(-1), v.reshape(-1), params)
        Ax = jac_constraints(x.reshape(-1), v.reshape(-1), params)
        Adot = Ax @ v.reshape(-1)
        xx = (AM_1 @ (-C @ v + Π + Υ + F) + Adot @ v)
        tmp = Aᵀ @ inv(AM_1 @ Aᵀ) @ xx
        out = M_1 @ (-C @ v + Π + Υ - tmp + F)
        return out.reshape(-1, Dim)
    return fn


def EL_parts(n, Dim, lagrangian=None, non_conservative_forces=None, external_force=None, constraints=None):
    """ ̈q = M⁻¹(-C ̇q + Π + Υ - Aᵀ(AM⁻¹Aᵀ)⁻¹ ( AM⁻¹ (-C ̇q + Π + Υ + F ) + Adot ̇q ) + F )

    :param T: [description], defaults to _T
    :type T: [type], optional
    :param lagrangian: [description], defaults to lagrangian
    :type lagrangian: [type], optional
    """
    def inv(x, *args, **kwargs):
        return jnp.linalg.pinv(x, *args, **kwargs)

    if non_conservative_forces == None:
        def non_conservative_forces(x, v, params): return 0
    if external_force == None:
        def external_force(x, v, params): return 0
    if constraints == None:
        def constraints(x, v, params): return jnp.zeros((1, n*Dim))

    eye = jnp.eye(n*Dim)

    def dL_dv(R, V, params):
        return jax.grad(lagrangian, 1)(R.reshape(n, Dim),
                                       V.reshape(n, Dim), params).flatten()

    def d2L_dv2(R, V, params):
        return jax.jacobian(dL_dv, 1)(R, V, params)
        # return eye*jnp.diag(jax.jacobian(dL_dv, 1)(R, V, params))

    def fn(x, v, params):
        N = n*Dim
        # M⁻¹ = (∂²L/∂²v)⁻¹
        M = d2L_dv2(x.flatten(), v.flatten(), params)
        M_1 = inv(M)
        # Π = ∂L/∂x
        Π = jax.grad(lagrangian, 0)(x, v, params).reshape(
            N, 1)
        # C = ∂²L/∂v∂x
        C = jax.jacobian(jax.jacobian(lagrangian, 1),
                         0)(x, v, params).reshape(N, N)

        Υ = non_conservative_forces(x, v, params)
        F = external_force(x, v, params)
        A = constraints(x.reshape(-1), v.reshape(-1), params)
        return M, M_1, Π, C, Υ, F, A, v, x

    return fn


def EL_from_parts(M, M_1, Π, C, Υ, F, A, v, x):
    """
    Aᵀ = A.T
    AM_1 = A @ M_1
    v = v.reshape(N, 1)
    Ax = jax.jacobian(constraints, 0)(x.reshape(-1), v.reshape(-1), None)
    Adot = Ax @ v.reshape(-1)
    xx = (AM_1 @ (-C @ v + Π + Υ + F) + Adot @ v)
    tmp = Aᵀ @ inv(AM_1 @ Aᵀ) @ xx
    out = M_1 @ (-C @ v + Π + Υ - tmp + F)
    """
    Aᵀ = A.T
    AM_1 = A @ M_1
    v = v.reshape(N, 1)
    Ax = jax.jacobian(constraints, 0)(x.reshape(-1), v.reshape(-1), params)
    Adot = Ax @ v.reshape(-1)
    xx = (AM_1 @ (-C @ v + Π + Υ + F) + Adot @ v)
    tmp = Aᵀ @ inv(AM_1 @ Aᵀ) @ xx
    out = M_1 @ (-C @ v + Π + Υ - tmp + F)
    return out.reshape(-1, Dim)


def accelerationModified(x, v, params):
    Dim = x.shape[1]
    N = x.shape[0]
    M_1 = forward_pass(params["M_1"], v.reshape(-1, ))
    M_1 = M_1.reshape(N*Dim, N*Dim)
    dx_L = jax.grad(lagrangian, 0)(x, v, params["PEF"]).reshape(-1, )
    dxdv_L = jax.jacobian(jax.jacobian(lagrangian, 1), 0)(
        x, v, params["PEF"]).reshape(N*Dim, N*Dim)
    F = (dx_L - dxdv_L @ v.reshape(-1, ))
    out = M_1 @ F
    return out.reshape(-1, Dim)


def force(x, v, params):
    dx_L = jax.grad(lagrangian, 0)(x, v, params)
    dxdv_L = jax.jacobian(jax.jacobian(lagrangian, 1), 0)(x, v, params)
    out = dx_L - dxdv_L @ v
    return out


def prediction_fn(X, params):
    x, v = jnp.split(X, 2)
    return acceleration(x, v, params)


# Make a batched version of the `predict` function
batch_prediction = vmap(prediction_fn, in_axes=(None, 0), out_axes=0)


# PEFs
# =============================================

def useNN(norm=True):
    """Create NNP function.

    :param norm: if take norm of input, defaults to True
    :type norm: bool, optional
    :return: NNP function
    :rtype: function
    """
    if norm:
        def f(x, params=None, cutoff=None):
            x_ = jnp.linalg.norm(x, keepdims=True)
            return jnp.where(x_ < cutoff, forward_pass(params, x_, activation_fn=SquarePlus), forward_pass(params, cutoff, activation_fn=SquarePlus))
        return f
    else:
        def f(x, params=None, cutoff=None):
            if cutoff is None:
                return forward_pass(params, x, activation_fn=SquarePlus)
            else:
                return jnp.where(x[-1] < cutoff, forward_pass(params, x, activation_fn=SquarePlus),
                                 forward_pass(params, jax.ops.index_update(x, -1, cutoff), activation_fn=SquarePlus))
        return f


def NNP(*args, **kwargs):
    """FFNN potential with cutoff.

    :param x: Inter-particle distance
    :type x: float
    :param params: NN parameters
    :type params: NN parameters
    :param cutoff: potential cutoff, defaults to None
    :type cutoff: float, optional
    :return: energy
    :rtype: float
    """
    return useNN()(*args, **kwargs)


def SPRING(x, stiffness=1.0, length=1.0):
    """Linear spring, v=0.5kd^2.

    :param x: Inter-particle distance
    :type x: float
    :param stiffness: Spring stiffness constant, defaults to 1.0
    :type stiffness: float, optional
    :param length: Equillibrium length, defaults to 1.0
    :type length: float, optional
    :return: energy
    :rtype: float
    """
    x_ = jnp.linalg.norm(x, keepdims=True)
    return 0.5*stiffness*(x_ - length)**2


def SPRING4(x, stiffness=1.0, length=1.0):
    """Non-linear spring, v=0.5kd^4.

    :param x: Inter-particle distance
    :type x: float
    :param stiffness: Spring stiffness constant, defaults to 1.0
    :type stiffness: float, optional
    :param length: Equillibrium length, defaults to 1.0
    :type length: float, optional
    :return: energy
    :rtype: float
    """
    x_ = jnp.linalg.norm(x, keepdims=True)
    return 0.5*stiffness*(x_ - length)**4


@ nonan
def GRAVITATIONAL(x, Gc=1.0):
    """Gravitational energy, Gc/r.

    :param x: Inter-particle distance.
    :type x: float
    :param Gc: Gravitational constant, defaults to 1.0
    :type Gc: float, optional
    :return: energy
    :rtype: float
    """
    x_ = jnp.linalg.norm(x, keepdims=True)
    return -Gc/x_


@ nonan
def VANDERWALLS(x, C=4.0):
    """Van Der Walls energy, C/r^12.

    :param x: Interatomic distance.
    :type x: float
    :param C: C, defaults to 4.0
    :type C: float, optional
    :return: energy
    :rtype: float
    """
    x_ = jnp.linalg.norm(x, keepdims=True)
    return C/x_**12


@ nonan
def x_6(x):
    """x^6

    :param x: value
    :type x: float
    :return: value
    :rtype: float
    """
    return 1.0/x**6


@ nonan
def x_3(x):
    """x^3

    :param x: value
    :type x: float
    :return: value
    :rtype: float
    """
    return 1.0/x**3


def LJ(x, sigma=1.0, epsilon=1.0):
    """Lennard-Jones (12-6) interatomic potential function.

    :param x: Interatomic distance
    :type x: float
    :param sigma: sigma, defaults to 1.0
    :type sigma: float, optional
    :param epsilon: epsilon, defaults to 1.0
    :type epsilon: float, optional
    :return: energy
    :rtype: float
    """
    x_ = jnp.sum(jnp.square(x), keepdims=True)
    r = x_3(x_)*sigma**6
    return 4.0*epsilon*(r**2 - r)

# =============================================


def t1(displacement=lambda a, b: a-b):
    """Create transformation function using displacement function.

    :param displacement: Dispalcement function to calculate euclidian displacemnt, defaults to lambda a, b: a - b
    :type displacement: Function, optional
    """
    def f(R):
        Dim = R.shape[1]
        # dd = displacement(R.reshape(-1, 1, Dim), R.reshape(1, -1, Dim))
        dd = vmap(vmap(displacement, in_axes=(0, None)),
                  in_axes=(None, 0))(R, R)
        indexs = jax.numpy.tril_indices(R.shape[0], k=-1)
        # R1, R2 = R[:20], R
        # dd = vmap(vmap(displacement, in_axes=(0, None)),
        #           in_axes=(None, 0))(R1, R2)
        # indexs = jax.numpy.triu_indices(R1.shape[0], 1, R2.shape[0])
        out = vmap(lambda i, j, dd: dd[i, j], in_axes=(
            0, 0, None))(indexs[0], indexs[1], dd)
        return out
    return f


def t2(q):
    """Apply transformation q -> q - q.mean(axis=0).

    :param q: Input array
    :type q: Array
    :return: Modified array
    :rtype: Array
    """
    q -= q.mean(axis=0, keepdims=True)
    return q


def t3(q):
    """No transformation.

    :param q: Input array.
    :type q: Array
    :return: Same as input.
    :rtype: Array
    """

    return q
# ================================


def cal_energy_parameters(params, states):
    kineticenergy = jnp.array([_T(state.velocity) for state in states])
    totallagrangian = jnp.array([lagrangian(state.position.reshape(-1,), state.velocity.reshape(-1,), params)
                                 for state in states])
    hamiltonian = 2*kineticenergy - totallagrangian
    return totallagrangian, hamiltonian, kineticenergy


def linear_mom_fn(states):
    return jnp.array([jnp.sqrt(jnp.square(state.velocity.sum(axis=0)).sum()) for state in states])


def angular_mom_fn(states):
    return jnp.array([jnp.sqrt(jnp.square(jnp.cross(state.position, state.velocity).sum(axis=0)).sum()) for state in states])
