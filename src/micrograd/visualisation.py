from dataclasses import dataclass

import graphviz  # pyright: ignore[reportMissingTypeStubs]
from micrograd.scalar import Scalar


@dataclass(frozen=True)
class Graph:
    nodes: set[Scalar]
    edges: set[tuple[Scalar, Scalar]]


def traverse(root: Scalar) -> Graph:
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
    graph = traverse(root)

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")  # pyright: ignore[reportUnknownMemberType]

    for node in graph.nodes:
        dot.node(  # pyright: ignore[reportUnknownMemberType]
            f"data_{id(node)}",
            shape="ellipse",
            label=f"data={node.data}, grad={node.grad}",
        )
        if node.op:
            dot.node(f"op_{id(node)}", shape="box", label=node.op)  # pyright: ignore[reportUnknownMemberType]
            dot.edge(f"op_{id(node)}", f"data_{id(node)}")  # pyright: ignore[reportUnknownMemberType]

    for parent, child in graph.edges:
        dot.edge(f"data_{id(parent)}", f"op_{id(child)}")  # pyright: ignore[reportUnknownMemberType]

    return dot
