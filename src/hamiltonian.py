import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import ode
from shadow.plot import panel


def hamiltonian(x, p, params):
    """
    hamiltonian calls lnn._H
    x: Vector
    p: Vector
    """
    return None


def ps(*args):
    for i in args:
        print(i.shape)


def get_zdot_lambda(N, Dim, hamiltonian, drag=None, constraints=None, external_force=None):
    dim = N*Dim
    I = jnp.eye(dim)
    J = jnp.zeros((2*dim, 2*dim))
    # J = jax.ops.index_update(J, jax.ops.index[:dim, dim:], I)
    # J = jax.ops.index_update(J, jax.ops.index[dim:, :dim], -I)
    J = J.at[:dim, dim:].set(I)
    J = J.at[dim:, :dim].set(-I)

    J2 = jnp.zeros((2*dim, 2*dim))
    # J2 = jax.ops.index_update(J2, jax.ops.index[:dim, :dim], I)
    # J2 = jax.ops.index_update(J2, jax.ops.index[dim:, dim:], I)
    J2 = J2.at[:dim, :dim].set(I)
    J2 = J2.at[dim:, dim:].set(I)

    def dH_dz(x, p, params):
        dH_dx = jax.grad(hamiltonian, 0)(x, p, params)
        dH_dp = jax.grad(hamiltonian, 1)(x, p, params)
        return jnp.hstack([dH_dx.flatten(), dH_dp.flatten()])

    if drag is None:
        def drag(x, p, params):
            return 0.0

    def dD_dz(x, p, params):
        dD_dx = jax.grad(drag, 0)(x, p, params)
        dD_dp = jax.grad(drag, 1)(x, p, params)
        return jnp.hstack([dD_dx.flatten(), dD_dp.flatten()])

    if external_force is None:
        def external_force(x, p, params):
            return 0.0*p

    if constraints is None:
        def constraints(x, p, params):
            return jnp.zeros((1, 2*dim))

    def fn_zdot(x, p, params):
        dH = dH_dz(x, p, params)
        dD = J2 @ dD_dz(x, p, params)
        dD = - J @ dD
        F = jnp.hstack(
            [jnp.zeros(dim), external_force(x, p, params).flatten()])
        F = -J @ F
        S = dH + J2 @ dD + F
        A = constraints(x, p, params).reshape(-1, 2*dim)
        Aᵀ = A.T
        INV = jnp.linalg.pinv(A @ J @ Aᵀ)
        λ = -INV @ A @ J @ S
        zdot = J @ (S + Aᵀ @ λ)
        return zdot.reshape(2*N, Dim)

    def fn_parts(x, p, params):
        dH = dH_dz(x, p, params)
        dD = J2 @ dD_dz(x, p, params)
        dD = - J @ dD
        F = jnp.hstack(
            [jnp.zeros(dim), external_force(x, p, params).flatten()])
        F = -J @ F
        S = dH + J2 @ dD + F
        A = constraints(x, p, params).reshape(-1, 2*dim)
        Aᵀ = A.T
        INV = jnp.linalg.pinv(A @ J @ Aᵀ)
        λ = -INV @ A @ J @ S
        zdot = J @ (S + Aᵀ @ λ)
        return dict(text="""
        dH = dH_dz(x, p, params)
        dD = J2 @ dD_dz(x, p, params)
        dD = - J @ dD
        F = jnp.hstack(
            [jnp.zeros(dim), external_force(x, p, params).flatten()])
        F = -J @ F
        S = dH + J2 @ dD + F
        A = constraints(x, p, params).reshape(-1, 2*dim)
        Aᵀ = A.T
        INV = jnp.linalg.pinv(A @ J @ Aᵀ)
        λ = -INV @ A @ J @ S
        zdot = J @ (S + Aᵀ @ λ)
        """, dH=dH, dD=dD, F=F, S=S, A=A, AT=Aᵀ, lambda_=λ, inv=INV, zdot=zdot)

    def lambda_force(x, p, params):
        dH = dH_dz(x, p, params)
        dD = J2 @ dD_dz(x, p, params)
        dD = - J @ dD
        F = jnp.hstack(
            [jnp.zeros(dim), external_force(x, p, params).flatten()])
        F = -J @ F
        S = dH + J2 @ dD + F
        A = constraints(x, p, params).reshape(-1, 2*dim)
        Aᵀ = A.T
        INV = jnp.linalg.pinv(A @ J @ Aᵀ)
        λ = -INV @ A @ J @ S
        return (J @ Aᵀ @ λ).reshape(2*N, Dim)
    return fn_zdot, lambda_force, fn_parts


def get_constraints(N, Dim, phi_, mass=None):
    if mass is None:
        mass = 1.0

    def phi(x): return phi_(x.reshape(N, Dim))

    def phidot(x, p):
        Dphi = jax.jacobian(phi)(x.flatten())
        pm = (p.flatten() / mass)
        return Dphi @ pm

    def psi(z):
        x, p = jnp.split(z, 2)
        return jnp.vstack([phi(x), phidot(x, p)])

    def Dpsi(z):
        return jax.jacobian(psi)(z)

    def fn(x, p, params):
        z = jnp.vstack([x, p])
        return Dpsi(z)

    return fn


def z(x, p): return jnp.vstack([x, p])

def _T(p, mass=jnp.array([1.0])):
    if len(mass) != len(p):
        mass = mass[0]*jnp.ones((len(p)))
    out = (1/mass)*jnp.square(p).sum(axis=1)
    return 0.5*out.sum()



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
