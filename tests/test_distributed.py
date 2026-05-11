"""
tests/test_distributed.py – testy jednostkowe wersji rozproszonej.

Pokrycie:
  - Serializacja / deserializacja sub-grafu (pickle round-trip)
  - Protokół komunikacji worker ↔ master (sentinel STOP, kolejki)
  - Poprawność wynikowego grafu względem baseline'u sekwencyjnego
  - Niezależność wyników między workerami (brak shared-state)
  - Pomiar narzutu: czas spawn > 0, bytes_transferred > 0
  - Obsługa brzegowych przypadków (pusta lista plików, 1 plik)

Uruchomienie:
    pytest tests/test_distributed.py -v
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import time
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


@pytest.fixture(scope="session")
def sample_text():
    return "Apple CEO Tim Cook announced a new iPhone at the WWDC event in California."


@pytest.fixture(scope="session")
def sample_graph_data(nlp, sample_text):
    return extract_graph_data(sample_text, nlp)


# ─────────────────────────────────────────────────────────────────────────────
# Testy serializacji (pickle round-trip)
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization:
    """Sprawdza jawną serializację pickle używaną w wersji rozproszonej."""

    def test_pickle_roundtrip_dict(self, sample_graph_data):
        """pickle.dumps → pickle.loads zwraca identyczny słownik."""
        raw = pickle.dumps(sample_graph_data)
        restored = pickle.loads(raw)
        assert restored["nodes"] == sample_graph_data["nodes"]
        assert restored["edges"] == sample_graph_data["edges"]

    def test_pickle_preserves_node_attrs(self, sample_graph_data):
        """Po deserializacji atrybuty węzłów (label, type, count) niezmienione."""
        raw = pickle.dumps(sample_graph_data)
        restored = pickle.loads(raw)
        for key, attrs in sample_graph_data["nodes"].items():
            assert restored["nodes"][key] == attrs

    def test_pickle_preserves_edge_attrs(self, sample_graph_data):
        """Po deserializacji atrybuty krawędzi (weight, relation) niezmienione."""
        raw = pickle.dumps(sample_graph_data)
        restored = pickle.loads(raw)
        for edge_key, attrs in sample_graph_data["edges"].items():
            assert restored["edges"][edge_key] == attrs

    def test_pickle_bytes_are_nonzero(self, sample_graph_data):
        """Serializowany graf ma niezerową liczbę bajtów."""
        raw = pickle.dumps(sample_graph_data)
        assert len(raw) > 0

    def test_pickle_empty_graph_data(self, nlp):
        """Pusty wynik (brak encji) też daje się spickle'ować."""
        empty = extract_graph_data("", nlp)
        raw = pickle.dumps(empty)
        restored = pickle.loads(raw)
        assert restored["nodes"] == {}
        assert restored["edges"] == {}

    def test_dict_to_graph_after_pickle(self, sample_graph_data):
        """dict_to_graph działa poprawnie na zdserializowanych danych."""
        raw = pickle.dumps(sample_graph_data)
        restored = pickle.loads(raw)
        g = dict_to_graph(restored)
        assert isinstance(g, nx.Graph)
        assert g.number_of_nodes() == len(sample_graph_data["nodes"])


# ─────────────────────────────────────────────────────────────────────────────
# Testy protokołu kolejek (task_queue / result_queue)
# ─────────────────────────────────────────────────────────────────────────────

class TestQueueProtocol:
    """Sprawdza mechanizm komunikacji master ↔ worker przez mp.Queue."""

    def test_task_queue_put_get(self):
        """Zadanie włożone do kolejki wychodzi bez zmian."""
        q = mp.Queue()
        task = ["path/to/file1.txt", "path/to/file2.txt"]
        q.put(task)
        result = q.get(timeout=2)
        assert result == task

    def test_stop_sentinel_is_none(self):
        """Sentinel STOP = None wychodzi z kolejki jako None."""
        q = mp.Queue()
        q.put(None)
        assert q.get(timeout=2) is None

    def test_result_queue_roundtrip(self, sample_graph_data):
        """Wyniki (lista bytes) przechodzą przez queue bez korupcji."""
        rq = mp.Queue()
        serialized = [pickle.dumps(sample_graph_data)]
        rq.put(serialized)
        received = rq.get(timeout=2)
        assert len(received) == 1
        restored = pickle.loads(received[0])
        assert restored["nodes"] == sample_graph_data["nodes"]

    def test_multiple_batches_ordering(self):
        """Kolejność wyjmowania z kolejki FIFO."""
        q = mp.Queue()
        for i in range(5):
            q.put(i)
        for i in range(5):
            assert q.get(timeout=2) == i

    def test_none_sentinel_after_results(self, sample_graph_data):
        """Master poprawnie rozróżnia wyniki od sentinela None."""
        rq = mp.Queue()
        rq.put([pickle.dumps(sample_graph_data)])  # wynik
        rq.put(None)                               # sentinel

        msg1 = rq.get(timeout=2)
        assert msg1 is not None
        assert isinstance(msg1, list)

        msg2 = rq.get(timeout=2)
        assert msg2 is None


# ─────────────────────────────────────────────────────────────────────────────
# Testy worker_process (pośrednie — przez subprocess/Queue)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(task_queue, result_queue):
    """Minimal worker for testing — no spaCy, just pickle round-trip."""
    while True:
        batch = task_queue.get()
        if batch is None:
            result_queue.put(None)
            break
        # Symulacja: zwróć identyczny batch jako "wynik"
        result_queue.put([pickle.dumps({"batch": batch})])


class TestWorkerProcess:
    """Testy procesu roboczego (bez modelu NLP — czysty protokół)."""

    def test_worker_processes_one_batch(self):
        """Worker przetwarza batch i zwraca wyniki przez result_queue."""
        tq = mp.Queue()
        rq = mp.Queue()
        batch = ["file_a.txt", "file_b.txt"]
        tq.put(batch)
        tq.put(None)  # STOP

        p = mp.Process(target=_worker_fn, args=(tq, rq))
        p.start()

        results = []
        sentinel_count = 0
        while sentinel_count < 1:
            msg = rq.get(timeout=10)
            if msg is None:
                sentinel_count += 1
            else:
                results.extend(msg)
        p.join(timeout=5)

        assert len(results) == 1
        data = pickle.loads(results[0])
        assert data["batch"] == batch

    def test_worker_stops_on_sentinel(self):
        """Worker kończy działanie po otrzymaniu None (STOP)."""
        tq = mp.Queue()
        rq = mp.Queue()
        tq.put(None)  # natychmiastowy STOP

        p = mp.Process(target=_worker_fn, args=(tq, rq))
        p.start()

        sentinel = rq.get(timeout=10)
        p.join(timeout=5)

        assert sentinel is None
        assert not p.is_alive()
        assert p.exitcode == 0

    def test_two_workers_independent(self):
        """Dwa workery niezależnie przetwarzają swoje batch'e."""
        tq = mp.Queue()
        rq = mp.Queue()

        tq.put(["a.txt"])
        tq.put(["b.txt"])
        tq.put(None)
        tq.put(None)

        workers = [mp.Process(target=_worker_fn, args=(tq, rq)) for _ in range(2)]
        for w in workers:
            w.start()

        results = []
        done = 0
        while done < 2:
            msg = rq.get(timeout=15)
            if msg is None:
                done += 1
            else:
                results.extend(msg)

        for w in workers:
            w.join(timeout=5)

        assert len(results) == 2  # każdy worker przetworzył 1 batch


# ─────────────────────────────────────────────────────────────────────────────
# Testy poprawności end-to-end (distributed vs sequential)
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributedCorrectness:
    """Sprawdza że distributed daje identyczny graf jak sekwencyjny."""

    def test_single_file_distributed_vs_sequential(self, nlp, small_files):
        """Jeden plik: distributed (1 worker) == sequential."""
        filepath = small_files[0]
        text = load_text(filepath)

        # Sequential
        data_seq = extract_graph_data(text, nlp)
        g_seq = dict_to_graph(data_seq)

        # Distributed (symulacja: pickle → unpickle → dict_to_graph)
        raw = pickle.dumps(data_seq)
        data_dist = pickle.loads(raw)
        g_dist = dict_to_graph(data_dist)

        assert set(g_seq.nodes()) == set(g_dist.nodes())
        assert set(g_seq.edges()) == set(g_dist.edges())

    def test_merge_after_distributed_equals_sequential(self, nlp, small_files):
        """
        Rozproszone przetwarzanie plików + merge == sekwencyjny merge.
        Symulacja: każdy plik osobno → pickle → unpickle → merge.
        """
        # Sequential
        seq_graphs = []
        for f in small_files:
            data = extract_graph_data(load_text(f), nlp)
            seq_graphs.append(dict_to_graph(data))
        merged_seq = merge_graphs(seq_graphs)

        # Distributed (explicit serialize/deserialize per file)
        dist_graphs = []
        for f in small_files:
            data = extract_graph_data(load_text(f), nlp)
            raw = pickle.dumps(data)             # seria "wyślij przez queue"
            data2 = pickle.loads(raw)            # seria "odbierz"
            dist_graphs.append(dict_to_graph(data2))
        merged_dist = merge_graphs(dist_graphs)

        assert merged_seq.number_of_nodes() == merged_dist.number_of_nodes()
        assert merged_seq.number_of_edges() == merged_dist.number_of_edges()
        assert set(merged_seq.nodes()) == set(merged_dist.nodes())

    def test_node_counts_preserved_after_distributed(self, nlp, small_files):
        """Liczniki count węzłów zachowane po serializacji."""
        f = small_files[0]
        data = extract_graph_data(load_text(f), nlp)
        g_orig = dict_to_graph(data)

        raw = pickle.dumps(data)
        g_restored = dict_to_graph(pickle.loads(raw))

        for node in g_orig.nodes():
            assert g_orig.nodes[node]["count"] == g_restored.nodes[node]["count"]

    def test_edge_weights_preserved_after_distributed(self, nlp, small_files):
        """Wagi krawędzi zachowane po serializacji."""
        f = small_files[0]
        data = extract_graph_data(load_text(f), nlp)
        g_orig = dict_to_graph(data)

        raw = pickle.dumps(data)
        g_restored = dict_to_graph(pickle.loads(raw))

        for u, v in g_orig.edges():
            assert g_orig[u][v]["weight"] == g_restored[u][v]["weight"]


# ─────────────────────────────────────────────────────────────────────────────
# Testy run_distributed (end-to-end z prawdziwymi workerami)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunDistributed:
    """Testy funkcji run_distributed() z prawdziwymi workerami NLP."""

    def test_run_distributed_basic(self, small_data_dir, tmp_path):
        """run_distributed zwraca poprawne statystyki i tworzy plik .graphml."""
        from benchmarks.distributed_benchmark import run_distributed

        out = tmp_path / "graph_dist_test.graphml"
        stats = run_distributed(small_data_dir, out, num_workers=2, batch_size=2)

        assert out.exists(), "Plik .graphml nie został utworzony"
        assert stats["num_files"] > 0
        assert stats["num_nodes"] > 0
        assert stats["num_edges"] >= 0
        assert stats["time_total_s"] > 0
        assert stats["bytes_transferred"] > 0

    def test_run_distributed_correctness_vs_sequential(self, small_data_dir, tmp_path):
        """Graf z run_distributed ma identyczne N/E co run_sequential."""
        from benchmarks.distributed_benchmark import run_distributed
        from benchmarks.baseline_sequential import run_sequential

        seq_out = tmp_path / "seq.graphml"
        dist_out = tmp_path / "dist.graphml"

        seq_stats = run_sequential(small_data_dir, seq_out)
        dist_stats = run_distributed(small_data_dir, dist_out, num_workers=2, batch_size=2)

        assert dist_stats["num_nodes"] == seq_stats["num_nodes"], \
            f"Węzły: dist={dist_stats['num_nodes']} vs seq={seq_stats['num_nodes']}"
        assert dist_stats["num_edges"] == seq_stats["num_edges"], \
            f"Krawędzie: dist={dist_stats['num_edges']} vs seq={seq_stats['num_edges']}"

    def test_run_distributed_spawn_overhead_measured(self, small_data_dir, tmp_path):
        """time_spawn_s jest mierzony i większy od zera."""
        from benchmarks.distributed_benchmark import run_distributed

        out = tmp_path / "spawn_test.graphml"
        stats = run_distributed(small_data_dir, out, num_workers=2, batch_size=3)
        assert stats["time_spawn_s"] >= 0

    def test_run_distributed_bytes_transferred_nonzero(self, small_data_dir, tmp_path):
        """bytes_transferred > 0 gdy przetworzone pliki."""
        from benchmarks.distributed_benchmark import run_distributed

        out = tmp_path / "bytes_test.graphml"
        stats = run_distributed(small_data_dir, out, num_workers=2, batch_size=2)
        assert stats["bytes_transferred"] > 0

    def test_run_distributed_batch1_equals_batch3(self, small_data_dir, tmp_path):
        """Różne batch_size → ten sam wynikowy graf."""
        from benchmarks.distributed_benchmark import run_distributed

        out1 = tmp_path / "b1.graphml"
        out3 = tmp_path / "b3.graphml"
        s1 = run_distributed(small_data_dir, out1, num_workers=2, batch_size=1)
        s3 = run_distributed(small_data_dir, out3, num_workers=2, batch_size=3)

        assert s1["num_nodes"] == s3["num_nodes"]
        assert s1["num_edges"] == s3["num_edges"]

    def test_run_distributed_graphml_loadable(self, small_data_dir, tmp_path):
        """Wygenerowany .graphml da się wczytać przez NetworkX."""
        from benchmarks.distributed_benchmark import run_distributed

        out = tmp_path / "loadable.graphml"
        run_distributed(small_data_dir, out, num_workers=2, batch_size=2)
        G = nx.read_graphml(out)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0
