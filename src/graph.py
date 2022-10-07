
"""A library of Graph Neural Network models."""

import functools
import sunau
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from frozendict import frozendict
from jax import vmap
from jraph._src import graph as gn_graph
from jraph._src import utils

from .models import SquarePlus, forward_pass

jax.tree_util.register_pytree_node(
    frozendict,
    flatten_func=lambda s: (tuple(s.values()), tuple(s.keys())),
    unflatten_func=lambda k, xs: frozendict(zip(k, xs)))

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray,
                  Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
# V: Potential energy of edge
GN_to_V_Fn = Callable[[EdgeFeatures, NodeFeatures], float]
GN_to_T_Fn = Callable[[NodeFeatures], float]


def GNNet(
    V_fn: GN_to_V_Fn,
    initial_edge_embed_fn: Optional[GNUpdateEdgeFn],
    initial_node_embed_fn: Optional[GNUpdateEdgeFn],
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    T_fn: GN_to_T_Fn = None,
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils
    .segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils
    .segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils
    .segment_softmax,
        attention_reduce_fn: Optional[AttentionReduceFn] = None,
        N=1,):
    """Returns a method that applies a configured GraphNetwork.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    than the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Example usage::

      gn = GraphNetwork(update_edge_function,
      update_node_function, **kwargs)
      # Conduct multiple rounds of message passing with the same parameters:
      for _ in range(num_message_passing_steps):
        graph = gn(graph)

    Args:
      update_edge_fn: function used to update the edges or None to deactivate edge
        updates.
      update_node_fn: function used to update the nodes or None to deactivate node
        updates.
      update_global_fn: function used to update the globals or None to deactivate
        globals updates.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
      aggregate_edges_for_globals_fn: function used to aggregate the edges for the
        globals.
      attention_logit_fn: function used to calculate the attention weights or
        None to deactivate attention mechanism.
      attention_normalize_fn: function used to normalize raw attention logits or
        None if attention mechanism is not active.
      attention_reduce_fn: function used to apply weights to the edge features or
        None if attention mechanism is not active.

    Returns:
      A method that applies the configured GraphNetwork.
    """
    def not_both_supplied(x, y): return (
        x != y) and ((x is None) or (y is None))
    if not_both_supplied(attention_reduce_fn, attention_logit_fn):
        raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                          ' supplied.'))

    def _ApplyGraphNet(graph):
        """Applies a configured GraphNetwork to a graph.

        This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

        There is one difference. For the nodes update the class aggregates over the
        sender edges and receiver edges separately. This is a bit more general
        the algorithm described in the paper. The original behaviour can be
        recovered by using only the receiver edge aggregations for the update.

        In addition this implementation supports softmax attention over incoming
        edge features.

        Many popular Graph Neural Networks can be implemented as special cases of
        GraphNets, for more information please see the paper.

        Args:
          graph: a `GraphsTuple` containing the graph.

        Returns:
          Updated `GraphsTuple`.


        """
        # pylint: disable=g-long-lambda
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        # Equivalent to jnp.sum(n_node), but jittable

        # calculate number of nodes in graph
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

        # calculate number of edges in graph
        sum_n_edge = senders.shape[0]

        # check if all all node array are of same length = number of nodes
        if not tree.tree_all(
                tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
            raise ValueError(
                'All node arrays in nest must contain the same number of nodes.')

        # Initial sent info
        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)

        # Initial received info
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)

        # Here we scatter the global features to the corresponding edges,
        # giving us tensors of shape [num_edges, global_feat].
        # i.e create an array per edge for global attributes
        global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

        # Here we scatter the global features to the corresponding nodes,
        # giving us tensors of shape [num_nodes, global_feat].
        # i.e create an array per node for global attributes
        global_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)

        # apply initial edge embeddings
        if initial_edge_embed_fn:
            edges = initial_edge_embed_fn(edges, sent_attributes, received_attributes,
                                          global_edge_attributes)
        # apply initial node embeddings
        if initial_node_embed_fn:
            nodes = initial_node_embed_fn(nodes, sent_attributes,
                                          received_attributes, global_attributes)

        # Now perform message passing for N times
        for pass_i in range(N):
            if attention_logit_fn:
                logits = attention_logit_fn(edges, sent_attributes, received_attributes,
                                            global_edge_attributes)
                tree_calculate_weights = functools.partial(
                    attention_normalize_fn,
                    segment_ids=receivers,
                    num_segments=sum_n_node)
                weights = tree.tree_map(tree_calculate_weights, logits)
                edges = attention_reduce_fn(edges, weights)

            if update_node_fn:
                nodes = update_node_fn(
                    nodes, edges, senders, receivers,
                    global_attributes, sum_n_node)

            if update_edge_fn:
                senders_attributes = tree.tree_map(
                    lambda n: n[senders], nodes)
                receivers_attributes = tree.tree_map(
                    lambda n: n[receivers], nodes)
                edges = update_edge_fn(edges, senders_attributes, receivers_attributes,
                                       global_edge_attributes, pass_i == N-1)

        if update_global_fn:
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            # To aggregate nodes and edges from each graph to global features,
            # we first construct tensors that map the node to the corresponding graph.
            # For example, if you have `n_node=[1,2]`, we construct the tensor
            # [0, 1, 1]. We then do the same for edges.
            node_gr_idx = jnp.repeat(
                graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
            edge_gr_idx = jnp.repeat(
                graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
            # We use the aggregation function to pool the nodes/edges per graph.
            node_attributes = tree.tree_map(
                lambda n: aggregate_nodes_for_globals_fn(
                    n, node_gr_idx, n_graph),
                nodes)
            edge_attribtutes = tree.tree_map(
                lambda e: aggregate_edges_for_globals_fn(
                    e, edge_gr_idx, n_graph),
                edges)
            # These pooled nodes are the inputs to the global update fn.
            globals_ = update_global_fn(
                node_attributes, edge_attribtutes, globals_)

        V = 0.0
        if V_fn is not None:
            V += V_fn(edges, nodes)

        T = 0.0
        if T_fn is not None:
            T += T_fn(nodes)

        # pylint: enable=g-long-lambda
        return gn_graph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge), V, T

    return _ApplyGraphNet


# Signature:
# edge features -> embedded edge features
EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]

# Signature:
# node features -> embedded node features
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]

# Signature:
# globals features -> embedded globals features
EmbedGlobalFn = Callable[[Globals], Globals]


def get_fully_connected_senders_and_receivers(
    num_particles: int, self_edges: bool = False,
):
    """Returns senders and receivers for fully connected particles."""
    particle_indices = np.arange(num_particles)
    senders, receivers = np.meshgrid(particle_indices, particle_indices)
    senders, receivers = senders.flatten(), receivers.flatten()
    if not self_edges:
        mask = senders != receivers
        senders, receivers = senders[mask], receivers[mask]
    return senders, receivers


def cal_graph(params, graph, eorder=None, mpass=1,
              useT=True, useonlyedge=False, act_fn=SquarePlus):
    fb_params = params["fb"]
    fne_params = params["fne"]
    fneke_params = params["fneke"]
    fv_params = params["fv"]
    fe_params = params["fe"]
    ff1_params = params["ff1"]
    ff2_params = params["ff2"]
    ff3_params = params["ff3"]
    ke_params = params["ke"]

    num_species = 1

    def onehot(n):
        def fn(n):
            out = jax.nn.one_hot(n, num_species)
            return out
        out = vmap(fn)(n.reshape(-1,))
        return out

    def fne(n):
        def fn(ni):
            out = forward_pass(fne_params, ni, activation_fn=lambda x: x)
            return out
        out = vmap(fn, in_axes=(0))(n)
        return out

    def fneke(n):
        def fn(ni):
            out = forward_pass(fneke_params, ni, activation_fn=lambda x: x)
            return out
        out = vmap(fn, in_axes=(0))(n)
        return out

    def fb(e):
        def fn(eij):
            out = forward_pass(fb_params, eij, activation_fn=act_fn)
            return out
        out = vmap(fn, in_axes=(0))(e)
        return out

    def fv(n, e, s, r, sum_n_node):
        c1ij = jnp.hstack([n[r], e])
        out = vmap(lambda x: forward_pass(fv_params, x))(c1ij)
        return n + jax.ops.segment_sum(out, r, sum_n_node)

    def fe(e, s, r):
        def fn(hi, hj):
            c2ij = hi * hj
            out = forward_pass(fe_params, c2ij, activation_fn=act_fn)
            return out
        out = e + vmap(fn, in_axes=(0, 0))(s, r)
        return out

    def ff1(e):
        def fn(eij):
            out = forward_pass(ff1_params, eij, activation_fn=act_fn)
            return out
        out = vmap(fn)(e)
        return out

    def ff2(n):
        def fn(ni):
            out = forward_pass(ff2_params, ni, activation_fn=act_fn)
            return out
        out = vmap(fn)(n)
        return out

    def ff3(n):
        def fn(ni):
            out = forward_pass(ff3_params, ni, activation_fn=act_fn)
            return out
        out = vmap(fn)(n)
        return out

    def ke(n):
        def fn(ni):
            out = forward_pass(ke_params, ni, activation_fn=act_fn)
            return out
        out = vmap(fn)(n)
        return out

    # ================================================================================

    def initial_edge_emb_fn(edges, senders, receivers, globals_):
        del edges, globals_
        dr = (senders["position"] - receivers["position"])
        # eij = dr
        eij = jnp.sqrt(jnp.square(dr).sum(axis=1, keepdims=True))
        emb = fb(eij)
        return frozendict({"edge_embed": emb, "eij": eij})

    def initial_node_emb_fn(nodes, sent_edges, received_edges, globals_):
        del sent_edges, received_edges, globals_
        type_of_node = nodes["type"]
        ohe = onehot(type_of_node)
        emb = fne(ohe)
        emb_pos = jnp.hstack([emb, nodes["position"]])
        emb_vel = jnp.hstack(
            [fneke(ohe), jnp.sum(jnp.square(nodes["velocity"]), axis=1, keepdims=True)])
        return frozendict({"node_embed": emb,
                           "node_pos_embed": emb_pos,
                           "node_vel_embed": emb_vel,
                           })

    def update_node_fn(nodes, edges, senders, receivers, globals_, sum_n_node):
        del globals_
        emb = fv(nodes["node_embed"], edges["edge_embed"],
                 senders, receivers, sum_n_node)
        n = dict(nodes)
        n.update({"node_embed": emb})
        return frozendict(n)

    def update_edge_fn(edges, senders, receivers, globals_, last_step):
        del globals_
        emb = fe(edges["edge_embed"], senders["node_embed"],
                 receivers["node_embed"])
        if last_step:
            if eorder is not None:
                emb = (emb + fe(edges["edge_embed"][eorder],
                       receivers["node_embed"], senders["node_embed"])) / 2
        return frozendict({"edge_embed": emb, "eij": edges["eij"]})

    if useonlyedge:
        def edge_node_to_V_fn(edges, nodes):
            vij = ff1(edges["edge_embed"])
            # print(vij, edges["eij"])
            return vij.sum()
    else:
        def edge_node_to_V_fn(edges, nodes):
            vij = ff1(edges["edge_embed"]).sum()
            vi = 0
            vi = vi + ff2(nodes["node_embed"]).sum()
            vi = vi + ff3(nodes["node_pos_embed"]).sum()
            return vij + vi

    def node_to_T_fn(nodes):
        return ke(nodes["node_vel_embed"]).sum()

    if not(useT):
        node_to_T_fn = None

    Net = GNNet(N=mpass,
                V_fn=edge_node_to_V_fn,
                T_fn=node_to_T_fn,
                initial_edge_embed_fn=initial_edge_emb_fn,
                initial_node_embed_fn=initial_node_emb_fn,
                update_edge_fn=update_edge_fn,
                update_node_fn=update_node_fn)

    return Net(graph)
