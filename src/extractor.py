"""
extractor.py – ekstrakcja encji i relacji z tekstu przy użyciu spaCy.

Węzły:  nazwane encje (NER) → (tekst, typ_encji)
Krawędzie:
  - co-occurrence within a sentence: encje współwystępujące w tym samym zdaniu
  - dependency relation: podmiot/dopełnienie czasownika (SVO triple)

Wagi krawędzi są zwiększane przy każdym kolejnym wystąpieniu pary.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import spacy

# Typy encji, które nas interesują
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"}


def _normalize(text: str) -> str:
    """Normalizuje tekst encji: strip + lower, usunięcie nadmiarowych spacji."""
    return " ".join(text.strip().lower().split())


def extract_graph_data(
    text: str,
    nlp: spacy.language.Language,
) -> dict[str, Any]:
    """
    Przetwarza jeden tekst i zwraca słownik:
      {
        'nodes': { normalized_label: {'label': str, 'type': str, 'count': int} },
        'edges': { (src, dst): {'relation': str, 'weight': int} },
      }
    Waga krawędzi = liczba współwystąpień lub relacji dep. w tym dokumencie.
    """
    doc = nlp(text)

    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = defaultdict(lambda: {"relation": "co-occurrence", "weight": 0})

    def add_node(ent_text: str, ent_label: str) -> str | None:
        if ent_label not in ENTITY_TYPES:
            return None
        key = _normalize(ent_text)
        if not key:
            return None
        if key not in nodes:
            nodes[key] = {"label": key, "type": ent_label, "count": 0}
        nodes[key]["count"] += 1
        return key

    def add_edge(src: str, dst: str, relation: str = "co-occurrence") -> None:
        if src == dst:
            return
        pair = (min(src, dst), max(src, dst))  # nieskierowane
        edges[pair]["weight"] += 1
        # Nadpisujemy relację jeśli jest bardziej konkretna niż co-occurrence
        if edges[pair]["relation"] == "co-occurrence" and relation != "co-occurrence":
            edges[pair]["relation"] = relation

    # 1. Węzły i krawędzie współwystępowania (w tym samym zdaniu)
    for sent in doc.sents:
        sent_ents = []
        for ent in sent.ents:
            key = add_node(ent.text, ent.label_)
            if key:
                sent_ents.append(key)

        # Każda para encji w zdaniu → krawędź co-occurrence
        for i in range(len(sent_ents)):
            for j in range(i + 1, len(sent_ents)):
                add_edge(sent_ents[i], sent_ents[j], "co-occurrence")

    # 2. Relacje dependency (SVO triples): subj–verb–obj
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            subj_ent = _get_entity_key(token, doc)
            for child in token.head.children:
                if child.dep_ in ("dobj", "pobj", "attr", "ccomp"):
                    obj_ent = _get_entity_key(child, doc)
                    if subj_ent and obj_ent and subj_ent in nodes and obj_ent in nodes:
                        relation = f"{token.head.lemma_}"
                        add_edge(subj_ent, obj_ent, relation)

    return {"nodes": nodes, "edges": dict(edges)}


def _get_entity_key(token: spacy.tokens.Token, doc: spacy.tokens.Doc) -> str | None:
    """Zwraca znormalizowany klucz encji dla tokena, jeśli należy do rozpoznanej encji."""
    if token.ent_type_ in ENTITY_TYPES:
        return _normalize(token.text)
    # Sprawdź czy token jest częścią dłuższej encji
    for ent in doc.ents:
        if ent.start <= token.i < ent.end and ent.label_ in ENTITY_TYPES:
            return _normalize(ent.text)
    return None
