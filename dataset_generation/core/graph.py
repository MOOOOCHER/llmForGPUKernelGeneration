from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TensorEdge:
    edge_id: str
    rank: int
    producer: Optional[str] = None
    consumers: List[str] = field(default_factory=list)
    is_program_input: bool = False
    create_op: str = "randn"


@dataclass
class OperatorNode:
    node_id: str
    op_name: str
    input_edges: List[str]
    output_edge: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_fusion_target: bool = False
    fusion_role: Optional[str] = None


@dataclass
class ProgramGraph:
    nodes: Dict[str, OperatorNode] = field(default_factory=dict)
    edges: Dict[str, TensorEdge] = field(default_factory=dict)
    output_edges: List[str] = field(default_factory=list)

    def add_input_edge(self, edge_id: str, rank: int, create_op: str = "randn") -> None:
        if edge_id in self.edges:
            raise ValueError(f"duplicate edge: {edge_id}")
        self.edges[edge_id] = TensorEdge(
            edge_id=edge_id,
            rank=rank,
            is_program_input=True,
            create_op=create_op,
        )

    def add_node(self, node: OperatorNode, output_rank: int) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node: {node.node_id}")
        if node.output_edge in self.edges:
            raise ValueError(f"duplicate output edge: {node.output_edge}")
        for edge_id in node.input_edges:
            if edge_id not in self.edges:
                raise ValueError(f"unknown input edge {edge_id} for {node.node_id}")
            self.edges[edge_id].consumers.append(node.node_id)
        self.nodes[node.node_id] = node
        self.edges[node.output_edge] = TensorEdge(
            edge_id=node.output_edge,
            rank=output_rank,
            producer=node.node_id,
        )

    def topological_nodes(self) -> List[OperatorNode]:
        # Nodes are inserted in topological order by the generator.
        return list(self.nodes.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.topological_nodes()],
            "edges": [asdict(edge) for edge in self.edges.values()],
            "outputs": list(self.output_edges),
        }
