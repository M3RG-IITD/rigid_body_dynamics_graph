"""
"""
import pickle

import jax.numpy as jnp


def loadfile(filename, verbose=False):
    if verbose:
        print(f"Loading {filename}")
    return pickle.load(open(filename, "rb"))


def savefile(filename, data, metadata={}, verbose=False):
    if verbose:
        print(f"Saving {filename}")
    pickle.dump((data, metadata), open(filename, "wb+"))


def save_ovito(filename, traj, species=None, lattice=None, length=None, insert_origin=False):
    """Save trajectory as ovito xyz file.

    Args:
        filename (string): File path.
        traj (list of states): Trajectory. 
    """
    print(f"Saving ovito file: {filename}")
    with open(filename, "w+") as ofile:
        for state in traj:
            N, dim = state.position.shape
            if species is None:
                species = jnp.array([1]*N).reshape(-1, 1)
            else:
                species = jnp.array(species).reshape(-1, 1)

            hinting = f"Properties=id:I:1:species:R:1:pos:R:{dim}:vel:R:{dim}:force:R:{dim}"
            tmp = jnp.eye(dim).flatten()
            if length is not None:
                lattice = " ".join(
                    [(f"{length}" if i != 0 else "0") for i in tmp])
                Lattice = f'Lattice="{lattice}"'
            if lattice is not None:
                Lattice = f'Lattice="{lattice}"'
            data = jnp.concatenate(
                [species, state.position, state.velocity, state.force], axis=1)
            if insert_origin:
                N = N + 1
                data = jnp.vstack([data, 0*data[1]])
            str_ = f"{N}" + f"\n{Lattice} {hinting}" + f" Time={state.time}\n"
            ofile.write(str_)
            for j in range(N):
                line = "\t".join([str(item) for item in data[j, :]])
                str_ = f"{j+1}\t" + line + "\n"
                ofile.write(str_)
