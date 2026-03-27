"""
pipeline.py – potok NLP dla jednego dokumentu.

Łączy loader, ekstraktor i konwerter do grafu.
Zaprojektowany tak, żeby łatwo było go wywołać z wersji równoległej
(każdy worker woła process_file() i zwraca częściowy graf).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import spacy
import networkx as nx

from src.extractor import extract_graph_data
from src.merge import dict_to_graph


def load_text(filepath: str | Path) -> str:
    """Wczytaj tekst z pliku .txt (UTF-8, z fallback na latin-1)."""
    p = Path(filepath)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def process_file(filepath: str | Path, nlp: spacy.language.Language) -> nx.Graph:
    """
    Przetwarza jeden plik tekstowy.
    Zwraca częściowy graf NetworkX (węzły + krawędzie z tego dokumentu).
    """
    text = load_text(filepath)
    data = extract_graph_data(text, nlp)
    return dict_to_graph(data)


def build_nlp(model: str = "en_core_web_sm") -> spacy.language.Language:
    """Ładuje i zwraca model spaCy. Wyłącza zbędne komponenty dla szybkości."""
    nlp = spacy.load(model, disable=["lemmatizer"])
    # Zwiększ limit znaków dla długich artykułów
    nlp.max_length = 2_000_000
    return nlp
