"""Visualisation utilities for automatic differentiation computation graphs.

This module provides utilities for creating visual representations of
scalar-based automatic differentiation computation graphs using Graphviz. The
visualisation shows data nodes as ellipses and operation nodes as boxes, with
directed edges representing dependencies between computations.
"""

from dataclasses import dataclass

import graphviz  # pyright: ignore[reportMissingTypeStubs]

from scalarflow import Scalar


@dataclass(frozen=True)
class Graph:
    """A computation graph representation for automatic differentiation.

    Attributes:
        nodes: The set of scalar nodes in the computation graph.
        edges: The set of directed edges representing dependencies between
           nodes. Each edge is a tuple of (parent, child) indicating data flow.
    """

    nodes: set[Scalar]
    edges: set[tuple[Scalar, Scalar]]


def traverse(root: Scalar) -> Graph:
    """Traverse a computation graph starting from a root scalar.

    This function performs a depth-first traversal of the computation graph
    starting from the given root scalar, collecting all nodes and edges that can
    be reached.

    Args:
        root: The scalar node to start traversal from.

    Returns:
        A Graph object containing all nodes and edges in the computation graph.
    """
    nodes: set[Scalar] = set()
    edges: set[tuple[Scalar, Scalar]] = set()
    stack = [root]
    while stack:
        nodes.add(node := stack.pop())
        for dep in node.deps:
            if dep not in nodes:
                stack.append(dep)
            edges.add((dep, node))
    return Graph(nodes, edges)


def visualise(root: Scalar) -> graphviz.Digraph:
    """Create a Graphviz visualisation of a computation graph.

    This function generates a visual representation of the automatic
    differentiation computation graph using Graphviz. Data nodes are displayed
    as ellipses with their values and gradients, while operation nodes are shown
    as boxes. The graph layout flows from left to right.

    Args:
        root: The scalar node to visualise the computation graph from.

    Returns:
        A Graphviz Digraph object that can be rendered or saved to visualise the
        computation graph.
    """
    graph = traverse(root)

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")  # pyright: ignore[reportUnknownMemberType]

    for node in graph.nodes:
        dot.node(  # pyright: ignore[reportUnknownMemberType]
            f"data_{id(node)}",
            shape="ellipse",
            label=f"data={node.data:.4f}, grad={node.grad:.4f}",
        )
        if node.op:
            dot.node(f"op_{id(node)}", shape="box", label=node.op)  # pyright: ignore[reportUnknownMemberType]
            dot.edge(f"op_{id(node)}", f"data_{id(node)}")  # pyright: ignore[reportUnknownMemberType]

    for parent, child in graph.edges:
        dot.edge(f"data_{id(parent)}", f"op_{id(child)}")  # pyright: ignore[reportUnknownMemberType]

    return dot
