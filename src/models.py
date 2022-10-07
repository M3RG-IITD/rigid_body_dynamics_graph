from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random, vmap

from .io import loadfile, savefile


def initialize_mlp(sizes, key, affine=[False], scale=1.0):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, len(sizes))
    # Initialize a single layer with Gaussian weights -  helper function
    if len(affine) != len(sizes):
        affine = [affine[0]]*len(sizes)
    affine[-1] = True

    def initialize_layer(m, n, key, affine=True, scale=1e-2):
        w_key, b_key = random.split(key)
        if affine:
            return scale * random.normal(w_key, (n, m)), 0 * random.normal(b_key, (n,))
        else:
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    return [initialize_layer(m, n, k, affine=a, scale=scale) for m, n, k, a in zip(sizes[:-1], sizes[1:], keys, affine)]


def SquarePlus(x):
    return lax.mul(0.5, lax.add(x, lax.sqrt(lax.add(lax.square(x), 4.))))


def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)


def layer(params, x):
    """ Simple ReLu layer for single sample """
    return jnp.dot(params[0], x) + params[1]


def forward_pass(params, x, activation_fn=SquarePlus):
    """ Compute the forward pass for each example individually """
    h = x

    # Loop over the ReLU hidden layers
    for p in params[:-1]:
        h = activation_fn(layer(p, h))

    # Perform final traformation
    p = params[-1]
    h = layer(p, h)
    return h

# Make a batched version of the `predict` function


def batch_forward(params, x, activation_fn=SquarePlus):
    return vmap(partial(forward_pass, activation_fn=activation_fn), in_axes=(None, 0), out_axes=0)(params, x)


def MSE(y_act, y_pred):
    return jnp.mean(jnp.square(y_pred - y_act))


def L2error(y_act, y_pred):
    return jnp.mean(jnp.square(y_pred - y_act))


def L1error(y_act, y_pred):
    return jnp.mean(jnp.abs(y_pred - y_act))


def batch_MSE(ys_act, ys_pred):
    return vmap(MSE, in_axes=(0, 0), out_axes=0)(ys_act, ys_pred)


def loadmodel(filename):
    model, metadata = loadfile(filename)
    if "multimodel" in metadata:
        params = {k: _makedictmodel(v) for k, v in model.items()}
    else:
        params = _makedictmodel(model)
    return params, metadata


def _makedictmodel(model):
    params = []
    for ind in range(len(model)):
        layer = model[f'layer_{ind}']
        w, b = layer["w"], layer["b"]
        params += [(w, b)]
    return params


def savemodel(filename, params, metadata={}):
    if type(params) is type({}):
        m = {k: _makemodeldict(v) for k, v in params.items()}
        metadata = {**metadata, "multimodel": True}
    else:
        m = _makemodeldict(params)
    savefile(filename, m, metadata=metadata)


def _makemodeldict(params):
    m = {}
    for ind, layer in enumerate(params):
        w, b = layer
        w_, b_ = jnp.array(w), jnp.array(b)
        m[f'layer_{ind}'] = {'w': w_, 'b': b_}
    return m


def _pprint_model(params, indent=""):
    for ind, layer in enumerate(params):
        w, b = layer
        print(
            f"{indent}#Layer {ind}: W ({w.shape}), b({b.shape}),  {w.shape[1]} --> {w.shape[0]}")


def pprint_model(params, Iindent=""):
    if type(params) != type({}):
        _pprint_model(params, indent=Iindent)
    else:
        for key, value in params.items():
            print(Iindent + ">" + key)
            indent = Iindent + "-"
            pprint_model(value, Iindent=indent)
