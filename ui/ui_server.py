"""
ui_server.py – FastAPI backend dla UI wizualizacji KG pipeline.

Endpointy:
  GET  /api/graph?dataset=small|medium   → węzły + krawędzie grafu (JSON)
  POST /api/run   { mode, dataset, workers, batch_size } → stats (JSON)
  GET  /api/files?dataset=small|medium   → lista plików

Uruchomienie:
    python ui_server.py
  otwórz http://localhost:8765
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── importy projektu ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import build_nlp, process_file
from src.extractor import extract_graph_data
from src.merge import merge_graphs, dict_to_graph
from benchmarks.baseline_sequential import run_sequential

app = FastAPI(title="KG Pipeline UI")

_NLP = None  # lazy load

def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = build_nlp()
    return _NLP


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _graph_to_json(G: nx.Graph) -> dict:
    nodes = [
        {
            "id": n,
            "label": d.get("label", n),
            "type": d.get("type", "UNK"),
            "count": d.get("count", 1),
        }
        for n, d in G.nodes(data=True)
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "weight": d.get("weight", 1),
            "relation": d.get("relation", "co-occurrence"),
        }
        for u, v, d in G.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def _data_dir(dataset: str) -> Path:
    p = Path(f"data/{dataset}")
    if not p.exists():
        raise HTTPException(404, f"Dataset '{dataset}' nie istnieje")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/files")
def list_files(dataset: str = "small"):
    d = _data_dir(dataset)
    files = sorted(d.glob("*.txt"))
    return {"files": [f.name for f in files], "count": len(files)}


@app.get("/api/graph")
def get_graph(dataset: str = "small"):
    d = _data_dir(dataset)
    nlp = get_nlp()
    files = sorted(d.glob("*.txt"))
    if not files:
        raise HTTPException(404, "Brak plików .txt")

    sub_graphs = [process_file(f, nlp) for f in files]
    G = merge_graphs(sub_graphs)
    return _graph_to_json(G)


class RunRequest(BaseModel):
    mode: str = "sequential"   # sequential | parallel | distributed
    dataset: str = "small"
    workers: int = 4
    batch_size: int = 2


@app.post("/api/run")
def run_pipeline(req: RunRequest):
    d = _data_dir(req.dataset)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()

    if req.mode == "sequential":
        out = out_dir / f"ui_seq_{req.dataset}.graphml"
        stats = run_sequential(d, out)
        stats.pop("data_dir", None)

    elif req.mode == "parallel":
        from benchmarks.parallel_benchmark import run_parallel
        out = out_dir / f"ui_par_{req.dataset}.graphml"
        stats = run_parallel(d, out, max_workers=req.workers, chunksize=req.batch_size)
        stats.pop("graph_obj", None)

    elif req.mode == "distributed":
        from benchmarks.distributed_benchmark import run_distributed
        out = out_dir / f"ui_dist_{req.dataset}.graphml"
        stats = run_distributed(d, out, num_workers=req.workers, batch_size=req.batch_size)
        stats.pop("graph_obj", None)

    else:
        raise HTTPException(400, f"Nieznany tryb: {req.mode}")

    # Dołącz graf do odpowiedzi
    G = nx.read_graphml(out)
    graph_data = _graph_to_json(G)

    return {
        "stats": stats,
        "graph": graph_data,
        "mode": req.mode,
        "dataset": req.dataset,
    }


@app.get("/api/results")
def list_results():
    out_dir = Path("results")
    csvs = list(out_dir.glob("*.csv"))
    data = []
    import csv
    for c in csvs:
        rows = []
        with open(c, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        data.append({"file": c.name, "rows": rows})
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Serwowanie UI (index.html)
# ─────────────────────────────────────────────────────────────────────────────

UI_PATH = Path(__file__).parent

@app.get("/", response_class=HTMLResponse)
def index():
    html = (UI_PATH / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


if __name__ == "__main__":
    print("KG Pipeline UI -> http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765, reload=False)
