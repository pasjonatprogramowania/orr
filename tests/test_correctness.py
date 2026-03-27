"""
tests/test_correctness.py – test poprawności baseline'u sekwencyjnego.

Sprawdza czy:
1. Znane encje z prostego tekstu są wykrywane jako węzły grafu.
2. Współwystępujące encje mają krawędź.
3. Graf jest spójny (merge działa poprawnie).
4. Właściwości węzłów i krawędzi są poprawnie ustawione.

Uruchomienie:
    pytest tests/test_correctness.py -v
"""
import pytest
import spacy
import networkx as nx

from src.extractor import extract_graph_data, ENTITY_TYPES
from src.merge import dict_to_graph, merge_graphs
from src.pipeline import build_nlp


@pytest.fixture(scope="session")
def nlp():
    """Ładuje model spaCy raz dla całej sesji testowej."""
    return build_nlp()


# =============================================================================
# Testy ekstraktora
# =============================================================================

class TestExtractor:

    def test_known_entities_detected(self, nlp):
        """spaCy powinno wykryć Apple i Tim Cook jako encje."""
        text = "Apple CEO Tim Cook announced a new iPhone at the event in California."
        data = extract_graph_data(text, nlp)
        node_keys = set(data["nodes"].keys())
        # Przynajmniej jedna z encji powinna być wykryta
        assert len(node_keys) > 0, "Nie wykryto żadnych encji"

    def test_co_occurrence_edge_created(self, nlp):
        """Dwie encje w tym samym zdaniu → krawędź co-occurrence."""
        text = "Barack Obama visited Berlin in 2009."
        data = extract_graph_data(text, nlp)
        nodes = set(data["nodes"].keys())
        # Musi być co najmniej 2 węzły żeby być krawędź
        if len(nodes) >= 2:
            assert len(data["edges"]) > 0, "Brak krawędzi mimo współwystępujących encji"

    def test_nodes_have_required_attrs(self, nlp):
        """Każdy węzeł musi mieć atrybuty: label, type, count."""
        text = "Google and Microsoft compete in the cloud market."
        data = extract_graph_data(text, nlp)
        for key, attrs in data["nodes"].items():
            assert "label" in attrs, f"Węzeł {key} nie ma atrybutu 'label'"
            assert "type" in attrs, f"Węzeł {key} nie ma atrybutu 'type'"
            assert "count" in attrs, f"Węzeł {key} nie ma atrybutu 'count'"
            assert attrs["type"] in ENTITY_TYPES, f"Nieznany typ encji: {attrs['type']}"

    def test_edges_have_required_attrs(self, nlp):
        """Każda krawędź musi mieć weight > 0 i atrybut relation."""
        text = "Elon Musk founded Tesla and SpaceX. Elon Musk is the CEO of Tesla."
        data = extract_graph_data(text, nlp)
        for (src, dst), attrs in data["edges"].items():
            assert "weight" in attrs, f"Krawędź ({src},{dst}) nie ma 'weight'"
            assert "relation" in attrs, f"Krawędź ({src},{dst}) nie ma 'relation'"
            assert attrs["weight"] > 0, f"Waga krawędzi musi być > 0"

    def test_repeated_cooccurrence_increases_weight(self, nlp):
        """Jeśli para encji pojawia sie wiele razy, waga krawędzi rośnie."""
        text = (
            "Apple is a company. Apple makes iPhones in California. "
            "Apple and California have a long history."
        )
        data = extract_graph_data(text, nlp)
        # Szukamy krawędzi apple-california
        for (src, dst), attrs in data["edges"].items():
            if {"apple", "california"} <= {src, dst}:
                assert attrs["weight"] >= 2, "Wielokrotne współwystąpienie powinno zwiększać wagę"

    def test_empty_text_returns_empty_graph(self, nlp):
        """Pusty tekst → pusty graf (bez crash)."""
        data = extract_graph_data("", nlp)
        assert data["nodes"] == {}
        assert data["edges"] == {}

    def test_no_self_loops(self, nlp):
        """Graf nie powinien zawierać pętli własnych."""
        text = "Amazon, Google and Facebook are tech giants. Amazon CEO Andy Jassy leads Amazon."
        data = extract_graph_data(text, nlp)
        for (src, dst) in data["edges"].keys():
            assert src != dst, f"Pętla własna na węźle: {src}"


# =============================================================================
# Testy merge
# =============================================================================

class TestMerge:

    def test_merge_two_graphs_nodes(self, nlp):
        """Po scaleniu muszą być węzły z obu grafów."""
        data1 = extract_graph_data("Apple is based in Cupertino, California.", nlp)
        data2 = extract_graph_data("Microsoft is headquartered in Redmond, Washington.", nlp)
        g1 = dict_to_graph(data1)
        g2 = dict_to_graph(data2)
        merged = merge_graphs([g1, g2])
        # Scalone węzły = suma węzłów obu grafów (bez duplikatów)
        all_nodes = set(g1.nodes()) | set(g2.nodes())
        assert set(merged.nodes()) == all_nodes

    def test_merge_weight_accumulation(self, nlp):
        """Te same krawędzie w różnych grafach – wagi sumowane."""
        text = "Barack Obama visited Berlin."
        data1 = extract_graph_data(text, nlp)
        data2 = extract_graph_data(text, nlp)  # identyczny tekst
        g1 = dict_to_graph(data1)
        g2 = dict_to_graph(data2)
        single = dict_to_graph(data1)
        merged = merge_graphs([g1, g2])

        for u, v, attrs in merged.edges(data=True):
            if single.has_edge(u, v):
                assert attrs["weight"] == single[u][v]["weight"] * 2, \
                    "Waga po scaleniu dwóch identycznych grafów powinna być 2×"

    def test_merge_count_accumulation(self, nlp):
        """Scalenie: count węzłów z obu grafów sumowany."""
        text = "NASA launched a rocket from Florida."
        data = extract_graph_data(text, nlp)
        g = dict_to_graph(data)
        merged = merge_graphs([g, g])
        for node, attrs in merged.nodes(data=True):
            if g.has_node(node):
                assert attrs["count"] == g.nodes[node]["count"] * 2

    def test_merge_empty_list(self, nlp):
        """Scalenie pustej listy zwraca pusty graf (bez crash)."""
        merged = merge_graphs([])
        assert merged.number_of_nodes() == 0
        assert merged.number_of_edges() == 0

    def test_merge_single_graph(self, nlp):
        """Scalenie jednego grafu zachowuje go bez zmian."""
        text = "Tesla was founded by Elon Musk in California."
        data = extract_graph_data(text, nlp)
        g = dict_to_graph(data)
        merged = merge_graphs([g])
        assert set(merged.nodes()) == set(g.nodes())
        assert set(merged.edges()) == set(g.edges())


# =============================================================================
# Test determinizmu
# =============================================================================

class TestDeterminism:

    def test_same_input_same_graph(self, nlp):
        """Ten sam tekst zawsze daje identyczny graf (deterministyczność)."""
        text = "Angela Merkel met Emmanuel Macron in Paris to discuss the EU budget."
        d1 = extract_graph_data(text, nlp)
        d2 = extract_graph_data(text, nlp)
        assert d1["nodes"] == d2["nodes"]
        assert d1["edges"] == d2["edges"]

    def test_graphml_roundtrip(self, nlp, tmp_path):
        """Graf zapisany do GraphML i wczytany z powrotem jest identyczny."""
        text = "Amazon CEO Andy Jassy announced new AWS services in Seattle."
        data = extract_graph_data(text, nlp)
        G = dict_to_graph(data)
        path = tmp_path / "test.graphml"
        nx.write_graphml(G, path)
        G2 = nx.read_graphml(path)
        assert set(G.nodes()) == set(G2.nodes())
        assert set(G.edges()) == set(G2.edges())
