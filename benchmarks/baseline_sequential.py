"""
baseline_sequential.py – sekwencyjny baseline ekstrakcji grafu wiedzy.

Uruchomienie:
    python baseline_sequential.py --data data/small --out results/graph_small.graphml
    python baseline_sequential.py --data data/medium --out results/graph_medium.graphml

Wynik:
    Plik .graphml z pełnym grafem wiedzy
    Statystyki (liczba węzłów, krawędzi, czas) wypisane na stdout i zapisane do results/
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from src.pipeline import build_nlp, process_file
from src.merge import merge_graphs


def run_sequential(data_dir: str | Path, out_path: str | Path) -> dict:
    """
    Sekwencyjnie przetwarza wszystkie pliki .txt z data_dir.
    Zapisuje wynikowy graf do out_path (.graphml).
    Zwraca słownik ze statystykami.
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Brak plików .txt w: {data_dir}")

    print(f"[baseline] Ładowanie modelu spaCy...")
    nlp = build_nlp()

    print(f"[baseline] Przetwarzanie {len(files)} plików z: {data_dir}")
    t_start = time.perf_counter()

    sub_graphs = []
    for f in tqdm(files, desc="processing", unit="doc"):
        g = process_file(f, nlp)
        sub_graphs.append(g)

    t_extract = time.perf_counter()

    print("[baseline] Scalanie sub-grafów...")
    final_graph = merge_graphs(sub_graphs)

    t_merge = time.perf_counter()
    t_total = t_merge - t_start

    # Zapis grafu
    nx.write_graphml(final_graph, out_path)
    print(f"[baseline] Graf zapisany: {out_path}")

    stats = {
        "data_dir": str(data_dir),
        "num_files": len(files),
        "num_nodes": final_graph.number_of_nodes(),
        "num_edges": final_graph.number_of_edges(),
        "time_extract_s": round(t_extract - t_start, 4),
        "time_merge_s": round(t_merge - t_extract, 4),
        "time_total_s": round(t_total, 4),
    }

    print("\n=== WYNIKI ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Sequential KG baseline")
    parser.add_argument("--data", required=True, help="Katalog z plikami .txt")
    parser.add_argument("--out", required=True, help="Ścieżka do pliku wynikowego .graphml")
    args = parser.parse_args()

    run_sequential(args.data, args.out)


if __name__ == "__main__":
    main()
