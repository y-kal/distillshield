from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from distillshield_core.schemas import SessionRecord


@dataclass
class OptionalModelOutput:
    available: bool
    score: float = 0.0
    details: dict | None = None


class SequenceModelStub:
    def __init__(self) -> None:
        self.available = torch is not None and nn is not None
        self.model = nn.LSTM(input_size=8, hidden_size=16, batch_first=True) if self.available else None

    def score(self, session: SessionRecord) -> OptionalModelOutput:
        if not self.available:
            return OptionalModelOutput(available=False, details={"reason": "PyTorch not installed"})
        score = min(len(session.queries) / 12.0, 1.0)
        return OptionalModelOutput(available=True, score=score, details={"stub": True})


class GraphModelStub:
    def build_graph(self, sessions: list[SessionRecord]) -> nx.Graph:
        graph = nx.Graph()
        for session in sessions:
            graph.add_node(session.user_id, kind="user")
            graph.add_node(session.id, kind="session")
            graph.add_node(session.api_context.api_key_id, kind="key")
            graph.add_node(session.api_context.ip_address, kind="ip")
            graph.add_edge(session.user_id, session.id)
            graph.add_edge(session.id, session.api_context.api_key_id)
            graph.add_edge(session.id, session.api_context.ip_address)
            graph.add_edge(session.id, session.api_context.org_id)
        return graph

    def score(self, session: SessionRecord, graph: nx.Graph | None = None) -> OptionalModelOutput:
        if graph is None:
            return OptionalModelOutput(available=False, details={"reason": "graph not provided"})
        degree = graph.degree(session.id) if session.id in graph else 0
        return OptionalModelOutput(available=True, score=min(degree / 6.0, 1.0), details={"graph_degree": degree, "gnn_stub": True})
