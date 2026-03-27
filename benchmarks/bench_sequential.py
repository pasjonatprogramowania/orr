"""
benchmarks/bench_sequential.py – benchmark harness dla wersji sekwencyjnej.

Mierzy czas przetwarzania dla data/small i data/medium.
Wyniki zapisuje do results/bench_sequential.csv i wypisuje na stdout.

Uruchomienie:
    python benchmarks/bench_sequential.py
    python benchmarks/bench_sequential.py --runs 3  # wielokrotne powtórzenia
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Dodaj root projektu do path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import build_nlp, process_file
from src.merge import merge_graphs


console = Console()


def benchmark_dir(data_dir: Path, nlp, runs: int = 3) -> dict:
    """
    Przeprowadza `runs` powtórzeń przetwarzania katalogu i zbiera statystyki.
    Zwraca słownik z wynikami.
    """
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        console.print(f"[red]Brak plików .txt w {data_dir}[/red]")
        return {}

    times_extract = []
    times_merge = []
    times_total = []
    num_nodes = None
    num_edges = None

    for run in range(1, runs + 1):
        t0 = time.perf_counter()

        sub_graphs = []
        for f in files:
            g = process_file(f, nlp)
            sub_graphs.append(g)

        t1 = time.perf_counter()

        merged = merge_graphs(sub_graphs)

        t2 = time.perf_counter()

        times_extract.append(t1 - t0)
        times_merge.append(t2 - t1)
        times_total.append(t2 - t0)

        # Zbieramy tylko raz
        if num_nodes is None:
            num_nodes = merged.number_of_nodes()
            num_edges = merged.number_of_edges()

        console.print(
            f"  Run {run}/{runs}: total={t2 - t0:.3f}s "
            f"(extract={t1 - t0:.3f}s, merge={t2 - t1:.3f}s)"
        )

    return {
        "dataset": data_dir.name,
        "num_files": len(files),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "runs": runs,
        "mean_total_s": round(statistics.mean(times_total), 4),
        "stdev_total_s": round(statistics.stdev(times_total) if runs > 1 else 0.0, 4),
        "min_total_s": round(min(times_total), 4),
        "max_total_s": round(max(times_total), 4),
        "mean_extract_s": round(statistics.mean(times_extract), 4),
        "mean_merge_s": round(statistics.mean(times_merge), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Sequential KG benchmark harness")
    parser.add_argument("--runs", type=int, default=3, help="Liczba powtórzeń (default: 3)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data/small", "data/medium"],
        help="Katalogi z danymi do zbenchmarkowania",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "bench_sequential.csv"

    console.print("[bold cyan]== Sequential KG Benchmark Harness ==[/bold cyan]")
    console.print(f"Ładowanie modelu spaCy...")
    nlp = build_nlp()

    all_results = []
    for ds in args.datasets:
        data_dir = root / ds
        console.print(f"\n[bold]Dataset: {data_dir}[/bold]")
        result = benchmark_dir(data_dir, nlp, runs=args.runs)
        if result:
            all_results.append(result)

    if not all_results:
        console.print("[red]Brak wyników – sprawdź katalogi z danymi.[/red]")
        return

    # Wyświetl tabelę wyników
    table = Table(title="Wyniki benchmarku – sekwencyjny baseline", show_lines=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Pliki", justify="right")
    table.add_column("Węzły", justify="right")
    table.add_column("Krawędzie", justify="right")
    table.add_column("Mean(s)", justify="right", style="green")
    table.add_column("StDev(s)", justify="right")
    table.add_column("Extract(s)", justify="right")
    table.add_column("Merge(s)", justify="right")

    for r in all_results:
        table.add_row(
            r["dataset"],
            str(r["num_files"]),
            str(r["num_nodes"]),
            str(r["num_edges"]),
            str(r["mean_total_s"]),
            str(r["stdev_total_s"]),
            str(r["mean_extract_s"]),
            str(r["mean_merge_s"]),
        )

    console.print(table)

    # Zapis CSV
    fieldnames = list(all_results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    console.print(f"\n[green]Wyniki zapisane: {out_csv}[/green]")


if __name__ == "__main__":
    main()
