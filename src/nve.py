
from typing import Callable, Tuple, TypeVar, Union

import jax.numpy as jnp
from jax import random
from jax_md import dataclasses, interpolate, quantity, simulate, space, util

static_cast = util.static_cast
# Types
Array = util.Array
f32 = util.f32
f64 = util.f64
ShiftFn = space.ShiftFn
T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]
Schedule = Union[Callable[..., float], float]


@dataclasses.dataclass
class NVEState:
    """A struct containing the state of an NVE simulation.
    This tuple stores the state of a simulation that samples from the
    microcanonical ensemble in which the (N)umber of particles, the (V)olume, and
    the (E)nergy of the system are held fixed.
    Attributes:
      position: An ndarray of shape [n, spatial_dimension] storing the position
        of particles.
      velocity: An ndarray of shape [n, spatial_dimension] storing the velocity
        of particles.
      force: An ndarray of shape [n, spatial_dimension] storing the force acting
        on particles from the previous step.
      mass: A float or an ndarray of shape [n] containing the masses of the
        particles.
      time: time
    """
    position: Array
    velocity: Array
    force: Array
    mass: Array
    time: Array


class NVEStates():
    def __init__(self, states):
        self.position = states.position
        self.velocity = states.velocity
        self.force = states.force
        self.mass = states.mass
        self.time = states.time
        self.index = 0

    def __len__(self):
        return len(self.position)

    def __getitem__(self, key):
        if isinstance(key, int):
            return NVEState(self.position[key],
                            self.velocity[key],
                            self.force[key],
                            self.mass[key],
                            self.time[key])
        else:
            return NVEState(self.position[key],
                            self.velocity[key],
                            self.force[key],
                            self.mass[key],
                            self.time[key])

    def __iter__(self,):
        return (self.__getitem__(i) for i in range(len(self)))


def nve(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float,
        dR_max: float) -> Simulator:
    """Simulates a system in the NVE ensemble.
    Samples from the microcanonical ensemble in which the number of particles
    (N), the system volume (V), and the energy (E) are held constant. We use a
    standard velocity verlet integration scheme.
    Args:
      energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        [n, spatial_dimension].
      shift_fn: A function that displaces positions, R, by an amount dR. Both R
        and dR should be ndarrays of shape [n, spatial_dimension].
      dt: Floating point number specifying the timescale (step size) of the
        simulation.
      quant: Either a quantity.Energy or a quantity.Force specifying whether
        energy_or_force is an energy or force respectively.
    Returns:
      See above.
    """
    force_fn = energy_or_force_fn
    dt_2 = 0.5 * dt ** 2

    def init_fun(R: Array,
                 V: Array,
                 mass=f32(1.0),
                 **kwargs) -> NVEState:
        mass = quantity.canonicalize_mass(mass)
        return NVEState(R, V, force_fn(R, V, **kwargs), mass, 0.0)

    def apply_fun(state: NVEState, **kwargs) -> NVEState:

        R, V, F, mass, time = dataclasses.astuple(state)
        A = F / mass

        # dR = V * dt + A * dt_2
        # k = dR_max / jnp.abs(dR).max()
        dt_ = dt #* jnp.where(k < 1.0, 0.1, 1.0)

        dR = V * dt_ + A * 0.5 * dt_**2
        R, V = shift_fn(R, dR, V)
        F = force_fn(R, V, **kwargs)
        A_prime = F / mass
        V = V + f32(0.5) * (A + A_prime) * dt_
        return NVEState(R, V, F, mass, time + dt_)

    return init_fun, apply_fun
