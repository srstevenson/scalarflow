from micrograd.scalar import Scalar
from micrograd.visualisation import Graph, traverse, visualise


def test__graph__init() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    nodes = {a, b}
    edges = {(a, b)}
    graph = Graph(nodes, edges)
    assert graph.nodes == nodes
    assert graph.edges == edges


def test__traverse__single_node() -> None:
    a = Scalar(1.0)
    graph = traverse(a)
    assert graph.nodes == {a}
    assert graph.edges == set()


def test__traverse__linear_chain() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = a + b
    graph = traverse(c)
    assert graph.nodes == {a, b, c}
    assert graph.edges == {(a, c), (b, c)}


def test__traverse__diamond_graph() -> None:
    a = Scalar(1.0)
    b = a + 2.0
    c = a * 3.0
    d = b + c
    graph = traverse(d)
    assert graph.nodes == {a, b, c, d}
    assert graph.edges == {(a, b), (a, c), (b, d), (c, d)}


def test__traverse__complex_expression() -> None:
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a * b
    d = c + 1.0
    e = d**2.0
    graph = traverse(e)
    assert graph.nodes == {a, b, c, d, e}
    assert graph.edges == {(a, c), (b, c), (c, d), (d, e)}


def test__traverse__nodes_completeness() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = Scalar(3.0)
    result = (a + b) * c
    graph = traverse(result)
    intermediate = next(dep for dep in result.deps if dep != c)
    assert graph.nodes == {a, b, c, intermediate, result}


def test__visualise__single_node() -> None:
    a = Scalar(1.0)
    dot = visualise(a)
    source = dot.source
    assert "ellipse" in source
    assert "1.0" in source


def test__visualise__linear_chain() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = a + b
    dot = visualise(c)
    assert "1.0" in dot.source
    assert "2.0" in dot.source
    assert "3.0" in dot.source
    assert dot.source.count("->") == 2


def test__visualise__complex_graph() -> None:
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a * b
    dot = visualise(c)
    assert "2.0" in dot.source
    assert "3.0" in dot.source
    assert "6.0" in dot.source


def test__visualise__node_attributes() -> None:
    a = Scalar(1.0)
    dot = visualise(a)
    assert "shape=ellipse" in dot.source
    assert "label=1.0" in dot.source


def test__visualise__edge_relationships() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = a + b
    graph = traverse(c)
    dot = visualise(c)
    assert dot.source.count("->") == len(graph.edges)


def test__visualise__rankdir_attribute() -> None:
    a = Scalar(1.0)
    dot = visualise(a)
    assert "rankdir=LR" in dot.source
