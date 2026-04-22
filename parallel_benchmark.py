import argparse
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from rich.console import Console
from rich.table import Table
import networkx as nx

from src.pipeline import build_nlp, process_file, load_text
from src.extractor import extract_graph_data
from src.merge import merge_graphs, dict_to_graph
from baseline_sequential import run_sequential

# Zmienna globalna w obrębie procesu-workera
_nlp = None

def init_worker():
    """Inicjalizacja modelu NLP osobno dla każdego procesu roboczego."""
    global _nlp
    _nlp = build_nlp()

def process_chunk(filepaths: list[str]) -> list[dict]:
    """
    Worker przyjmuje listę ścieżek do plików, wyciąga tekst i korzysta z lokalnej
    instancji modelu _nlp do wyciągnięcia uproszczonych relacji i węzłów.
    Zwraca lekkie słowniki (dict), aby minimalizować narzut na IPC.
    """
    global _nlp
    if _nlp is None:
        raise RuntimeError("Model NLP nie został zainicjalizowany w workerze!")
    
    results = []
    for path in filepaths:
        text = load_text(path)
        data = extract_graph_data(text, _nlp)
        results.append(data)
    return results

def run_parallel(data_dir: Path, out_path: Path, max_workers: int, chunksize: int) -> dict:
    """
    Uruchamia równoległe przetwarzanie plików tekstowych, tworzy grafy częściowe i scala je.
    """
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Brak plików w {data_dir}")

    # Tworzymy porcje (chunks) dla workerów
    chunks = [files[i:i + chunksize] for i in range(0, len(files), chunksize)]

    t_start = time.perf_counter()

    extracted_dicts_list = []
    
    # Proces ekstrakcji równoległej
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        for chunk_results in executor.map(process_chunk, chunks):
            extracted_dicts_list.extend(chunk_results)

    t_extract = time.perf_counter()

    # Proces konwersji lekkich wyników na grafy i scalanie sekwencyjne
    sub_graphs = [dict_to_graph(d) for d in extracted_dicts_list]
    final_graph = merge_graphs(sub_graphs)

    t_merge = time.perf_counter()

    # Zapis grafu
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(final_graph, out_path)

    stats = {
        "num_files": len(files),
        "num_nodes": final_graph.number_of_nodes(),
        "num_edges": final_graph.number_of_edges(),
        "time_extract_s": round(t_extract - t_start, 4),
        "time_merge_s": round(t_merge - t_extract, 4),
        "time_total_s": round(t_merge - t_start, 4),
        "graph_obj": final_graph # uzyte do weryfikacji w petli
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Parallel KG MVP Benchmark")
    parser.add_argument("--data", required=True, help="Katalog z plikami .txt")
    parser.add_argument("--out-dir", default="results", help="Katalog zapisu wyników")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"\n--- Start MVP Parallel Benchmark ---")
    print(f"Dane wejściowe: {data_dir}")
    
    # KROK 1: Wykonanie i pomiar wersji sekwencyjnej (Baseline)
    print("\n[Uruchamianie Baseline (Sekwencyjnie)...]")
    baseline_out = out_dir / "graph_baseline.graphml"
    
    # Przechwytujemy standardowe stdout baselinu lub robimy wlasne logi (wrapper)
    # Ze względu na budowę baseline_sequential.run_sequential - ono printuje swoje dzialanie.
    baseline_stats = run_sequential(data_dir, baseline_out)
    baseline_time = baseline_stats["time_total_s"]
    expected_nodes = baseline_stats["num_nodes"]
    expected_edges = baseline_stats["num_edges"]

    # KROK 2: Testy Równoległe (Różne konfiguracje)
    workers_list = [2, 4, 8]
    chunk_sizes = [1, 2, 5, 10]
    
    # Dodanie max_workers w zależności od CPU
    max_cpu = mp.cpu_count()
    if max_cpu not in workers_list:
        workers_list.append(max_cpu)
        
    results_table = []
    
    print("\n[Uruchamianie Testów Równoległych...]")
    for w in set(workers_list):
        for c in chunk_sizes:
            # Pomiń nieopłacalne konfiguracje, jeśli jest bardzo mało plików (tu opcjonalne, puszczamy wszystko)
            test_name = f"{w}W_{c}C"
            out_file = out_dir / f"graph_parallel_{test_name}.graphml"
            
            print(f" -> Test: {w} Workerów, Chunk: {c}... ", end="", flush=True)
            stats = run_parallel(data_dir, out_file, max_workers=w, chunksize=c)
            
            # Weryfikacja
            g = stats.pop("graph_obj")
            nodes_match = g.number_of_nodes() == expected_nodes
            edges_match = g.number_of_edges() == expected_edges
            is_correct = "Zgodny" if (nodes_match and edges_match) else "BŁĄD!"
            
            speedup = baseline_time / stats["time_total_s"]
            
            results_table.append([
                w, c, 
                stats["time_extract_s"], 
                stats["time_merge_s"], 
                stats["time_total_s"], 
                f"{speedup:.2f}x",
                is_correct
            ])
            print(f"Gotowe (Czas: {stats['time_total_s']}s, {is_correct})")

    # KROK 3: Podsumowanie wyników w formie czytelnej tabeli
    console = Console()
    console.print("\n[bold cyan]=== RAPORT Z WYDAJNOŚCI ===[/bold cyan]")
    console.print("[bold]Dane odniesienia (Baseline - 1 wątek):[/bold]")
    console.print(f" - Czas ekstrakcji: {baseline_stats['time_extract_s']}s")
    console.print(f" - Czas scalania: {baseline_stats['time_merge_s']}s")
    console.print(f" - Całkowity czas: {baseline_time}s")
    console.print(f" - Wygenerowany graf: Węzłów: {expected_nodes}, Krawędzi: {expected_edges}\n")
    console.print("[bold]Porównanie konfiguracji wielowątkowych:[/bold]")
    
    # Sortowanie tabeli rosnąco po Total [s] (indeks 4) aby wyłonić najlepszego
    results_table.sort(key=lambda x: x[4])
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Workery", justify="right")
    table.add_column("Chunksize", justify="right")
    table.add_column("Extrakcja [s]", justify="right")
    table.add_column("Scalanie [s]", justify="right")
    table.add_column("Total [s]", justify="right")
    table.add_column("Przyspieszenie", justify="right", style="green")
    table.add_column("Zgodność z Baseline", justify="center")

    for row in results_table:
        table.add_row(
            str(row[0]), str(row[1]), str(row[2]), str(row[3]), 
            str(row[4]), str(row[5]), str(row[6])
        )
        
    console.print(table)
    
    # Zapis wyników do pliku CSV
    headers = ["Workery", "Chunksize", "Extrakcja [s]", "Scalanie [s]", "Total [s]", "Przyspieszenie", "Zgodność z Baseline"]
    csv_path = out_dir / "parallel_benchmark_results.csv"
    import csv
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results_table)
    console.print(f"[bold green] \nWyniki zostały pomyślnie zapisane do pliku: {csv_path}[/bold green]")
    
if __name__ == "__main__":
    main()
