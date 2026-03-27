"""
merge.py – scalanie sub-grafów w jeden centralny graf NetworkX.

Strategia:
  - węzły z tym samym kluczem są scalane (count sumowany)
  - krawędzie między tymi samymi węzłami: wagi sumowane
  - relacja: preferujemy bardziej konkretną (nie 'co-occurrence') jeśli dostępna
"""
from __future__ import annotations

import networkx as nx


def dict_to_graph(data: dict) -> nx.Graph:
    """Konwertuje wynik extract_graph_data() na obiekt nx.Graph."""
    G = nx.Graph()
    for key, attrs in data["nodes"].items():
        G.add_node(key, **attrs)
    for (src, dst), attrs in data["edges"].items():
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst, **attrs)
    return G


def merge_graphs(graphs: list[nx.Graph]) -> nx.Graph:
    """
    Scala listę grafów częściowych w jeden spójny graf.
    Węzły: suma count z każdego sub-grafu.
    Krawędzie: suma weight; relacja = najbardziej konkretna napotkana.
    """
    merged = nx.Graph()

    for G in graphs:
        # Węzły
        for node, attrs in G.nodes(data=True):
            if merged.has_node(node):
                merged.nodes[node]["count"] += attrs.get("count", 1)
            else:
                merged.add_node(node, **attrs)

        # Krawędzie
        for u, v, attrs in G.edges(data=True):
            # Ignoruj krawędzie, których węzły nie istnieją w scalonym grafie
            if not merged.has_node(u) or not merged.has_node(v):
                continue
            if merged.has_edge(u, v):
                merged[u][v]["weight"] += attrs.get("weight", 1)
                # Preferuj konkretniejszą relację
                existing_rel = merged[u][v].get("relation", "co-occurrence")
                new_rel = attrs.get("relation", "co-occurrence")
                if existing_rel == "co-occurrence" and new_rel != "co-occurrence":
                    merged[u][v]["relation"] = new_rel
            else:
                merged.add_edge(u, v, **attrs)

    return merged
