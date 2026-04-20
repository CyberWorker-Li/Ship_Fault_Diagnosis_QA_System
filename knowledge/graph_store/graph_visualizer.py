from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import networkx as nx
from pyvis.network import Network


@dataclass
class RenderOptions:
    max_nodes: int = 150
    max_edges: int = 250
    min_importance: int = 1
    show_edge_labels: bool = False
    enable_physics: bool = False


class GraphVisualizer:
    def __init__(self, options: RenderOptions | None = None):
        self.graph = nx.DiGraph()
        self.options = options or RenderOptions()

    def build(
        self,
        triples: Iterable[Tuple[str, str, str]],
        seed_nodes: Sequence[str] | None = None,
        max_hops: int = 1,
    ) -> None:
        self.graph.clear()
        importance = defaultdict(int)
        relation_support = defaultdict(int)

        for s, r, o in triples:
            if self.graph.has_edge(s, o):
                self.graph[s][o]["support"] = int(self.graph[s][o].get("support", 1)) + 1
            else:
                self.graph.add_edge(s, o, relation=r, support=1)
            importance[s] += 1
            importance[o] += 1
            relation_support[r] += 1

        nx.set_node_attributes(self.graph, dict(importance), "importance")
        for s, o, attrs in self.graph.edges(data=True):
            attrs["relation_global_support"] = relation_support.get(attrs.get("relation", ""), 1)

        if seed_nodes:
            self.graph = self._extract_neighborhood(seed_nodes, max_hops=max_hops)

        self._trim_graph()

    def _extract_neighborhood(self, seeds: Sequence[str], max_hops: int = 1) -> nx.DiGraph:
        seeds = [s for s in (seeds or []) if s in self.graph]
        if not seeds:
            return self.graph
        u = self.graph.to_undirected()
        keep = set(seeds)
        for s in seeds:
            lengths = nx.single_source_shortest_path_length(u, s, cutoff=max(1, int(max_hops)))
            keep.update(lengths.keys())
        return self.graph.subgraph(keep).copy()

    def _trim_graph(self) -> None:
        if self.options.min_importance > 1:
            keep = [n for n, d in self.graph.nodes(data=True) if int(d.get("importance", 1)) >= self.options.min_importance]
            self.graph = self.graph.subgraph(keep).copy()

        if self.graph.number_of_nodes() > self.options.max_nodes:
            ranked = sorted(
                self.graph.nodes(),
                key=lambda n: int(self.graph.nodes[n].get("importance", 1)) + self.graph.degree(n),
                reverse=True,
            )
            self.graph = self.graph.subgraph(ranked[: self.options.max_nodes]).copy()

        if self.graph.number_of_edges() > self.options.max_edges:
            ranked_edges = sorted(
                self.graph.edges(data=True),
                key=lambda e: (
                    int(e[2].get("support", 1)),
                    int(self.graph.nodes[e[0]].get("importance", 1)) + int(self.graph.nodes[e[1]].get("importance", 1)),
                ),
                reverse=True,
            )
            keep_edges = ranked_edges[: self.options.max_edges]
            g = nx.DiGraph()
            for n, attrs in self.graph.nodes(data=True):
                g.add_node(n, **attrs)
            for s, o, attrs in keep_edges:
                g.add_edge(s, o, **attrs)
            self.graph = g

    def render_html(self, output_file: str) -> None:
        net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#222", directed=True)
        net.set_options(
            """
            {
              "interaction": {"hover": true, "hideEdgesOnDrag": true},
              "physics": {"enabled": false}
            }
            """
            if not self.options.enable_physics
            else '{"interaction":{"hover":true},"physics":{"enabled":true}}'
        )

        importance = nx.get_node_attributes(self.graph, "importance")
        for node in self.graph.nodes():
            imp = int(importance.get(node, 1))
            title = f"{node} | 重要性: {imp} | 入度: {self.graph.in_degree(node)} | 出度: {self.graph.out_degree(node)}"
            net.add_node(node, label=node, title=title, size=16 + min(imp, 10) * 2)

        for s, o, attrs in self.graph.edges(data=True):
            rel = str(attrs.get("relation", "关联"))
            if self.options.show_edge_labels:
                net.add_edge(s, o, label=rel, title=rel)
            else:
                net.add_edge(s, o, title=rel)

        net.save_graph(output_file)