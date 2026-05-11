"""
distributed_benchmark.py – distributed-like wersja ekstrakcji grafu wiedzy.

Architektura master-worker z jawnymi kolejkami zadań i wyników:
  - MASTER  : czyta pliki, wysyła ZADANIA (ścieżki plików) przez task_queue
              odbiera WYNIKI (serializowane słowniki sub-grafów) przez result_queue
              i scala je w jeden spójny graf końcowy.
  - WORKER  : nasłuchuje na task_queue, ładuje własną instancję modelu NLP (raz!),
              przetwarza kolejne pliki aż do otrzymania sygnału STOP,
              odsyła zserializowany sub-graf przez result_queue.

Kluczowa cecha odróżniająca od wersji równoległej (ProcessPoolExecutor):
  - Jawna, asymetryczna komunikacja przez kolejki (IPC / simulation of network).
  - Każdy worker jest osobnym procesem z własnym PID i cyklem życia.
  - Brak wspólnej pamięci – dane są ZAWSZE serializowane (pickle) przed wysłaniem.
  - Sygnał STOP (sentinel) wysyłany do każdego workera osobno, symulując EOF kanału.
  - Możliwe rozszerzenie na prawdziwe gniazda TCP (multiprocessing.managers).

Uruchomienie:
    python distributed_benchmark.py --data data/medium --workers 4
    python distributed_benchmark.py --data data/medium --workers 4 --batch-size 3
    python distributed_benchmark.py --data data/medium --all-configs
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import pickle
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Any

import networkx as nx
from rich.console import Console
from rich.table import Table

from src.pipeline import build_nlp, load_text
from src.extractor import extract_graph_data
from src.merge import merge_graphs, dict_to_graph
from .baseline_sequential import run_sequential

# ─────────────────────────────────────────────────────────────────────────────
# Protokół komunikacji (sentinel values)
# ─────────────────────────────────────────────────────────────────────────────
_STOP = None  # sentinel wysyłany do workera, gdy brak już zadań


# ─────────────────────────────────────────────────────────────────────────────
# WORKER process
# ─────────────────────────────────────────────────────────────────────────────

def worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Pętla workera:
      1. Jednorazowo ładuje model NLP (koszt inicjalizacji ponoszony raz).
      2. W pętli pobiera BATCH ścieżek plików z task_queue.
      3. Przetwarza każdy plik i serializuje wynik (dict).
      4. Wysyła wyniki przez result_queue.
      5. Kończy po odebraniu sygnału STOP.

    Serializacja odbywa się jawnie przez pickle.dumps / pickle.loads,
    aby zademonstrować narzut komunikacyjny (symulacja warstwy sieciowej).
    """
    nlp = build_nlp()
    processed = 0

    while True:
        batch = task_queue.get()          # blokuje do momentu dostępności
        if batch is _STOP:
            # Sygnał zakończenia – wyślij None jako sentinel do mastera
            result_queue.put(None)
            break

        batch_results: list[bytes] = []
        for filepath in batch:
            text = load_text(filepath)
            data = extract_graph_data(text, nlp)
            # Jawna serializacja – symulacja przesyłu przez sieć
            serialized = pickle.dumps(data)
            batch_results.append(serialized)
            processed += 1

        # Wyślij całą paczkę wyników (kolejne pickle)
        result_queue.put(batch_results)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER / scheduler
# ─────────────────────────────────────────────────────────────────────────────

def run_distributed(
    data_dir: Path,
    out_path: Path,
    num_workers: int,
    batch_size: int,
) -> dict[str, Any]:
    """
    Uruchamia potok distributed-like:
      Master → task_queue → Workers → result_queue → Master (merge)

    Mierzy osobno:
      - czas startu workerów
      - czas dystrybucji zadań i ekstrakcji (end-to-end, master czeka)
      - czas serializacji / deserializacji (szacunkowy)
      - czas scalania sub-grafów
    """
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Brak plików .txt w: {data_dir}")

    # Podział na batch'e zadań
    batches: list[list[Path]] = [
        files[i: i + batch_size] for i in range(0, len(files), batch_size)
    ]

    # ── 1. Start workerów ──────────────────────────────────────────────────
    t0_spawn = time.perf_counter()

    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    workers: list[mp.Process] = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(wid, task_queue, result_queue),
            name=f"Worker-{wid}",
            daemon=True,
        )
        p.start()
        workers.append(p)

    t_spawned = time.perf_counter()
    time_spawn_s = round(t_spawned - t0_spawn, 4)

    # ── 2. Dystrybucja zadań (master → worker) ────────────────────────────
    t_send_start = time.perf_counter()
    for batch in batches:
        # Konwertuj Path → str przed wysłaniem (pickle-friendly)
        task_queue.put([str(p) for p in batch])

    # Wyślij sygnał STOP do każdego workera
    for _ in workers:
        task_queue.put(_STOP)

    t_sent = time.perf_counter()

    # ── 3. Zbieranie wyników (worker → master) ────────────────────────────
    serialized_results: list[bytes] = []
    done_workers = 0

    while done_workers < num_workers:
        msg = result_queue.get()
        if msg is None:
            done_workers += 1
        else:
            serialized_results.extend(msg)

    t_received = time.perf_counter()
    time_extract_s = round(t_received - t_send_start, 4)

    # ── 4. Deserializacja + scalanie ──────────────────────────────────────
    t_merge_start = time.perf_counter()

    sub_graphs: list[nx.Graph] = []
    bytes_transferred = 0

    for raw in serialized_results:
        bytes_transferred += len(raw)
        data = pickle.loads(raw)           # jawna deserializacja
        sub_graphs.append(dict_to_graph(data))

    final_graph = merge_graphs(sub_graphs)

    t_merge_end = time.perf_counter()
    time_merge_s = round(t_merge_end - t_merge_start, 4)
    time_total_s = round(t_merge_end - t0_spawn, 4)

    # ── 5. Cleanup ────────────────────────────────────────────────────────
    for p in workers:
        p.join(timeout=10)

    # Zapis grafu
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(final_graph, out_path)

    return {
        "num_files": len(files),
        "num_batches": len(batches),
        "num_workers": num_workers,
        "batch_size": batch_size,
        "num_nodes": final_graph.number_of_nodes(),
        "num_edges": final_graph.number_of_edges(),
        "time_spawn_s": time_spawn_s,
        "time_extract_s": time_extract_s,
        "time_merge_s": time_merge_s,
        "time_total_s": time_total_s,
        "bytes_transferred": bytes_transferred,
        "graph_obj": final_graph,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Force UTF-8 stdout on Windows only when running as CLI (not when imported by tests)
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except AttributeError:
            pass  # already wrapped or not a real file
    parser = argparse.ArgumentParser(description="Distributed-like KG Benchmark")
    parser.add_argument("--data", required=True, help="Katalog z plikami .txt")
    parser.add_argument("--out-dir", default="results", help="Katalog zapisu wyników")
    parser.add_argument("--workers", type=int, default=4, help="Liczba workerów")
    parser.add_argument("--batch-size", type=int, default=2, help="Rozmiar batch'a zadań")
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Testuj wszystkie kombinacje workers × batch-size",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    console = Console()
    console.print("\n[bold cyan]=== DISTRIBUTED-LIKE BENCHMARK ===[/bold cyan]")
    console.print(f"Dane wejściowe: [bold]{data_dir}[/bold]\n")

    # ── Baseline ─────────────────────────────────────────────────────────────
    console.print("[bold]>> Uruchamianie Baseline (sekwencyjny)...[/bold]")
    baseline_out = out_dir / "graph_baseline_dist.graphml"
    baseline_stats = run_sequential(data_dir, baseline_out)
    baseline_time = baseline_stats["time_total_s"]
    expected_nodes = baseline_stats["num_nodes"]
    expected_edges = baseline_stats["num_edges"]

    console.print(
        f"  Baseline: {baseline_time}s | "
        f"Węzły: {expected_nodes} | Krawędzie: {expected_edges}\n"
    )

    # ── Konfiguracje ─────────────────────────────────────────────────────────
    if args.all_configs:
        max_cpu = mp.cpu_count()
        workers_list = sorted({2, 4, 8, max_cpu})
        batch_sizes = [1, 2, 5]
    else:
        workers_list = [args.workers]
        batch_sizes = [args.batch_size]

    results_table: list[list] = []

    for n_workers in workers_list:
        for b_size in batch_sizes:
            label = f"{n_workers}W_{b_size}B"
            out_file = out_dir / f"graph_distributed_{label}.graphml"

            console.print(
                f" -> [{label}] Workers={n_workers}, BatchSize={b_size}... ",
                end="",
            )
            stats = run_distributed(data_dir, out_file, n_workers, b_size)

            # Weryfikacja poprawności
            g = stats.pop("graph_obj")
            nodes_ok = g.number_of_nodes() == expected_nodes
            edges_ok = g.number_of_edges() == expected_edges
            correctness = "[OK] Zgodny" if (nodes_ok and edges_ok) else "[BLAD!]"

            speedup = baseline_time / stats["time_total_s"] if stats["time_total_s"] > 0 else float("inf")
            kb_transferred = round(stats["bytes_transferred"] / 1024, 1)

            console.print(
                f"[green]{stats['time_total_s']}s[/green] | "
                f"Spawn: {stats['time_spawn_s']}s | "
                f"Przesłano: {kb_transferred} KB | {correctness}"
            )

            results_table.append([
                n_workers,
                b_size,
                stats["num_batches"],
                stats["time_spawn_s"],
                stats["time_extract_s"],
                stats["time_merge_s"],
                stats["time_total_s"],
                f"{speedup:.2f}x",
                f"{kb_transferred} KB",
                correctness,
            ])

    # ── Tabela wynikowa ──────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]=== RAPORT DISTRIBUTED ===[/bold cyan]")
    console.print(f"[bold]Baseline: {baseline_time}s | Węzły: {expected_nodes} | Krawędzie: {expected_edges}[/bold]\n")

    results_table.sort(key=lambda r: r[6])  # sort by total time

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Workers", justify="right")
    table.add_column("BatchSize", justify="right")
    table.add_column("Batche", justify="right")
    table.add_column("Spawn [s]", justify="right")
    table.add_column("Ekstrakcja [s]", justify="right")
    table.add_column("Scalanie [s]", justify="right")
    table.add_column("Total [s]", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("Przesłano [KB]", justify="right")
    table.add_column("Zgodność", justify="center")

    for row in results_table:
        table.add_row(*[str(x) for x in row])

    console.print(table)

    # ── Zapis CSV ───────────────────────────────────────────────────────────
    headers = [
        "Workers", "BatchSize", "Batche",
        "Spawn [s]", "Ekstrakcja [s]", "Scalanie [s]", "Total [s]",
        "Speedup", "Przesłano [KB]", "Zgodność z Baseline",
    ]
    csv_path = out_dir / "distributed_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results_table)

    console.print(f"\n[bold green]Wyniki zapisane do: {csv_path}[/bold green]")


if __name__ == "__main__":
    mp.freeze_support()  # wymagane na Windows dla spawnowanych procesów
    main()
