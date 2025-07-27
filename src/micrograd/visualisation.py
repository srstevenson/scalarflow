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
        for child in node.children:
            if child not in nodes:
                stack.append(child)
            edges.add((child, node))
    return Graph(nodes, edges)


def visualise(root: Scalar) -> graphviz.Digraph:
    graph = traverse(root)

    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")  # pyright: ignore[reportUnknownMemberType]

    for node in graph.nodes:
        dot.node(str(id(node)), shape="ellipse", label=f"{node.data}")  # pyright: ignore[reportUnknownMemberType]

    for child, parent in graph.edges:
        dot.edge(str(id(child)), str(id(parent)))  # pyright: ignore[reportUnknownMemberType]

    return dot
