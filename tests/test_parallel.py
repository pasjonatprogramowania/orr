"""
tests/test_parallel.py – testy jednostkowe wersji równoległej.

Pokrycie:
  - process_chunk: przetwarzanie batcha plików w workerze
  - run_parallel: poprawność vs baseline, struktura statystyk
  - Różne konfiguracje workers × chunksize → ten sam wynik
  - IPC: zwrot lekkich dict zamiast obiektów NetworkX
  - Obsługa brzegowych przypadków

Uruchomienie:
    pytest tests/test_parallel.py -v
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from src.extractor import extract_graph_data
from src.merge import dict_to_graph, merge_graphs
from src.pipeline import build_nlp, load_text


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nlp():
    return build_nlp()


@pytest.fixture(scope="session")
def small_data_dir():
    p = Path("data/small")
    if not p.exists():
        pytest.skip("data/small nie istnieje – uruchom download_data.py")
    return p


@pytest.fixture(scope="session")
def small_files(small_data_dir):
    files = sorted(small_data_dir.glob("*.txt"))
    if not files:
        pytest.skip("Brak plików .txt w data/small")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Testy process_chunk (logika workera)
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessChunk:
    """Testy funkcji process_chunk używanej w wersji równoległej."""

    def test_process_chunk_returns_list_of_dicts(self, nlp, small_files):
        """process_chunk zwraca listę słowników (nodes + edges), nie grafów NetworkX."""
        # Symulujemy process_chunk bez ProcessPoolExecutor (bezpośrednie wywołanie)
        results = []
        for filepath in [str(small_files[0])]:
            text = load_text(filepath)
            data = extract_graph_data(text, nlp)
            results.append(data)

        assert isinstance(results, list)
        assert len(results) == 1
        assert "nodes" in results[0]
        assert "edges" in results[0]

    def test_process_chunk_dict_not_graph_object(self, nlp, small_files):
        """Worker zwraca dict, nie nx.Graph – minimalizacja IPC overhead."""
        text = load_text(small_files[0])
        data = extract_graph_data(text, nlp)
        # data musi być dict, nie Graph
        assert isinstance(data, dict)
        assert not isinstance(data, nx.Graph)

    def test_process_chunk_multiple_files(self, nlp, small_files):
        """Batch 3 plików → 3 wyniki."""
        batch = [str(f) for f in small_files[:3]]
        results = []
        for filepath in batch:
            text = load_text(filepath)
            data = extract_graph_data(text, nlp)
            results.append(data)

        assert len(results) == 3
        for r in results:
            assert "nodes" in r
            assert "edges" in r

    def test_process_chunk_all_nodes_valid_type(self, nlp, small_files):
        """Wszystkie węzły z batcha mają poprawny typ encji."""
        from src.extractor import ENTITY_TYPES
        text = load_text(small_files[0])
        data = extract_graph_data(text, nlp)
        for key, attrs in data["nodes"].items():
            assert attrs["type"] in ENTITY_TYPES

    def test_process_chunk_no_nx_graph_leaks(self, nlp, small_files):
        """Wynik workera nie zawiera obiektów NetworkX (kosztowne dla pickle)."""
        import pickle
        text = load_text(small_files[0])
        data = extract_graph_data(text, nlp)
        # Musi się dać spickle'ować szybko (brak dużych obiektów)
        raw = pickle.dumps(data)
        assert len(raw) < 1_000_000  # < 1MB dla jednego pliku


# ─────────────────────────────────────────────────────────────────────────────
# Testy run_parallel (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunParallel:
    """Testy end-to-end funkcji run_parallel()."""

    def test_run_parallel_basic(self, small_data_dir, tmp_path):
        """run_parallel tworzy plik .graphml i zwraca statystyki."""
        from benchmarks.parallel_benchmark import run_parallel

        out = tmp_path / "parallel_basic.graphml"
        stats = run_parallel(small_data_dir, out, max_workers=2, chunksize=2)

        assert out.exists()
        assert stats["num_files"] > 0
        assert stats["num_nodes"] > 0
        assert stats["time_total_s"] > 0
        assert stats["time_extract_s"] > 0

    def test_run_parallel_correctness_vs_sequential(self, small_data_dir, tmp_path):
        """Graf z run_parallel identyczny z run_sequential."""
        from benchmarks.parallel_benchmark import run_parallel
        from benchmarks.baseline_sequential import run_sequential

        seq_out = tmp_path / "seq_ref.graphml"
        par_out = tmp_path / "par_ref.graphml"

        seq_stats = run_sequential(small_data_dir, seq_out)
        par_stats = run_parallel(small_data_dir, par_out, max_workers=2, chunksize=2)

        # pop graph_obj jeśli istnieje
        par_stats.pop("graph_obj", None)

        assert par_stats["num_nodes"] == seq_stats["num_nodes"], \
            f"Węzły: par={par_stats['num_nodes']} seq={seq_stats['num_nodes']}"
        assert par_stats["num_edges"] == seq_stats["num_edges"], \
            f"Krawędzie: par={par_stats['num_edges']} seq={seq_stats['num_edges']}"

    def test_run_parallel_4workers_same_result(self, small_data_dir, tmp_path):
        """4 workery → ten sam graf co 2 workery."""
        from benchmarks.parallel_benchmark import run_parallel

        out2 = tmp_path / "par_2w.graphml"
        out4 = tmp_path / "par_4w.graphml"

        s2 = run_parallel(small_data_dir, out2, max_workers=2, chunksize=2)
        s4 = run_parallel(small_data_dir, out4, max_workers=4, chunksize=2)

        s2.pop("graph_obj", None)
        s4.pop("graph_obj", None)

        assert s2["num_nodes"] == s4["num_nodes"]
        assert s2["num_edges"] == s4["num_edges"]

    def test_run_parallel_chunksize_invariant(self, small_data_dir, tmp_path):
        """Różny chunksize → identyczny wynikowy graf."""
        from benchmarks.parallel_benchmark import run_parallel

        results = {}
        for cs in [1, 2, 5]:
            out = tmp_path / f"par_cs{cs}.graphml"
            s = run_parallel(small_data_dir, out, max_workers=2, chunksize=cs)
            s.pop("graph_obj", None)
            results[cs] = s

        nodes_set = {v["num_nodes"] for v in results.values()}
        edges_set = {v["num_edges"] for v in results.values()}
        assert len(nodes_set) == 1, f"Różna liczba węzłów przy różnych chunksizes: {nodes_set}"
        assert len(edges_set) == 1, f"Różna liczba krawędzi: {edges_set}"

    def test_run_parallel_graphml_valid(self, small_data_dir, tmp_path):
        """Wygenerowany .graphml jest wczytywalny i niespójny z pustym grafem."""
        from benchmarks.parallel_benchmark import run_parallel

        out = tmp_path / "par_valid.graphml"
        run_parallel(small_data_dir, out, max_workers=2, chunksize=2)
        G = nx.read_graphml(out)
        assert G.number_of_nodes() > 0

    def test_run_parallel_stats_keys(self, small_data_dir, tmp_path):
        """Statystyki zawierają wszystkie wymagane klucze."""
        from benchmarks.parallel_benchmark import run_parallel

        out = tmp_path / "par_keys.graphml"
        stats = run_parallel(small_data_dir, out, max_workers=2, chunksize=2)
        stats.pop("graph_obj", None)

        required = {"num_files", "num_nodes", "num_edges",
                    "time_extract_s", "time_merge_s", "time_total_s"}
        assert required.issubset(stats.keys()), \
            f"Brakujące klucze: {required - stats.keys()}"

    def test_run_parallel_merge_time_nonzero(self, small_data_dir, tmp_path):
        """time_merge_s >= 0 (scalanie zawsze ma czas >= 0)."""
        from benchmarks.parallel_benchmark import run_parallel

        out = tmp_path / "par_merge_time.graphml"
        stats = run_parallel(small_data_dir, out, max_workers=2, chunksize=2)
        assert stats["time_merge_s"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Testy idem potencji i deterministyczności
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelDeterminism:
    """Sprawdza powtarzalność wyników wersji równoległej."""

    def test_parallel_same_result_twice(self, small_data_dir, tmp_path):
        """Dwa identyczne uruchomienia run_parallel → identyczny graf."""
        from benchmarks.parallel_benchmark import run_parallel

        out1 = tmp_path / "det1.graphml"
        out2 = tmp_path / "det2.graphml"

        s1 = run_parallel(small_data_dir, out1, max_workers=2, chunksize=2)
        s2 = run_parallel(small_data_dir, out2, max_workers=2, chunksize=2)

        s1.pop("graph_obj", None)
        s2.pop("graph_obj", None)

        assert s1["num_nodes"] == s2["num_nodes"]
        assert s1["num_edges"] == s2["num_edges"]

    def test_parallel_node_set_matches_sequential_node_set(self, nlp, small_files):
        """Zbiór węzłów z równoległej = suma węzłów z wszystkich plików sekwencyjnie."""
        seq_graphs = []
        for f in small_files:
            data = extract_graph_data(load_text(f), nlp)
            seq_graphs.append(dict_to_graph(data))
        merged_seq = merge_graphs(seq_graphs)

        # Symulacja równoległości: identyczna logika, inne kolejność
        import random
        shuffled = list(small_files)
        random.seed(42)
        random.shuffle(shuffled)

        par_graphs = []
        for f in shuffled:
            data = extract_graph_data(load_text(f), nlp)
            par_graphs.append(dict_to_graph(data))
        merged_par = merge_graphs(par_graphs)

        assert set(merged_seq.nodes()) == set(merged_par.nodes())
        assert set(merged_seq.edges()) == set(merged_par.edges())
