from collections import defaultdict
from typing import Iterable, Tuple

import networkx as nx
from pyvis.network import Network


class GraphVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, triples: Iterable[Tuple[str, str, str]]) -> None:
        self.graph.clear()
        importance = defaultdict(int)
        for s, r, o in triples:
            self.graph.add_edge(s, o, relation=r)
            importance[s] += 1
            importance[o] += 1
        nx.set_node_attributes(self.graph, dict(importance), "importance")

    def render_html(self, output_file: str) -> None:
        net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#222", directed=True)
        importance = nx.get_node_attributes(self.graph, "importance")
        for node in self.graph.nodes():
            imp = importance.get(node, 1)
            net.add_node(node, label=node, title=f"{node} | 重要性: {imp}", size=20 + min(imp, 8) * 2)
        for s, o, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "关联")
            net.add_edge(s, o, label=rel, title=rel)
        net.save_graph(output_file)