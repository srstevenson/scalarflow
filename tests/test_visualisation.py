from scalarflow.scalar import Scalar
from scalarflow.visualisation import Graph, traverse, visualise


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
    assert len(graph.nodes) == 6

    for node in [a, b, c, d]:
        assert node in graph.nodes

    primitive_values = {node.data for node in graph.nodes if not node.deps}
    assert 2.0 in primitive_values
    assert 3.0 in primitive_values


def test__traverse__complex_expression() -> None:
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a * b
    d = c + 1.0
    e = d**2.0
    graph = traverse(e)

    assert len(graph.nodes) == 7
    for node in [a, b, c, d, e]:
        assert node in graph.nodes

    primitive_values = {node.data for node in graph.nodes if not node.deps}
    assert 1.0 in primitive_values
    assert 2.0 in primitive_values


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
    assert dot.source.count("->") == 3


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
    assert 'label="data=1.0000, grad=0.0000"' in dot.source


def test__visualise__edge_relationships() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = a + b
    graph = traverse(c)
    dot = visualise(c)
    # We have graph.edges, plus 1 (op, data) edge
    assert dot.source.count("->") == len(graph.edges) + 1


def test__visualise__rankdir_attribute() -> None:
    a = Scalar(1.0)
    dot = visualise(a)
    assert "rankdir=LR" in dot.source


def test__visualise__operation_nodes() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = a + b
    dot = visualise(c)
    source = dot.source
    assert 'label="+"' in source
    assert "shape=box" in source


def test__visualise__operation_node_shapes() -> None:
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a * b
    dot = visualise(c)
    source = dot.source
    assert 'label="×"' in source
    assert "shape=box" in source
    assert "shape=ellipse" in source


def test__visualise__operation_to_data_edges() -> None:
    a = Scalar(1.0)
    b = Scalar(2.0)
    c = a + b
    dot = visualise(c)
    # We have an (op_id, data_id) edge
    assert "op_" in dot.source
    assert "data_" in dot.source


def test__visualise__multiple_operations() -> None:
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = a * b
    d = c + 1.0
    dot = visualise(d)
    assert 'label="×"' in dot.source
    assert 'label="+"' in dot.source
    assert dot.source.count("shape=box") == 2
    assert dot.source.count("shape=ellipse") >= 3


def test__visualise__power_operation() -> None:
    a = Scalar(2.0)
    b = a**3.0
    dot = visualise(b)
    assert 'label="^"' in dot.source
    assert "shape=box" in dot.source
