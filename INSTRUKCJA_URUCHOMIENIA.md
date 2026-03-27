# Instrukcja uruchomienia projektu

Poniższa instrukcja krok po kroku wyjaśnia, jak pobrać dane, przygotować środowisko i uruchomić wersję sekwencyjną, testy oraz benchmark.

## Krok 1: Przygotowanie środowiska

Projekt wymaga Pythona w wersji 3.10+ (zalecane 3.12).
Otwórz terminal (PowerShell / Command Prompt) w folderze `ORR` na pulpicie i uruchom poniższe komendy:

1. Utworzenie wirtualnego środowiska:

   ```powershell
   python -m venv venv
   ```

2. Aktywacja wirtualnego środowiska:

   ```powershell
   # Na Windows:
   .\venv\Scripts\activate
   ```

3. Instalacja wszystkich wymaganych bibliotek:

   ```powershell
   pip install -r requirements.txt
   ```

4. Pobranie i instalacja modelu językowego dla `spaCy` (wymagane do przetwarzania NLP):

   ```powershell
   python -m spacy download en_core_web_sm
   ```

---

## Krok 2: Pobranie i przygotowanie danych wejściowych

Skrypt samodzielnie pobiera 1.6 GB corpus tekstów z seriwsu Kaggle i podziale go na małe (`small`) oraz średnie (`medium`) zestawy testowe:

```powershell
python download_data.py
```

*Oczekiwany efekt: Zostaną pobrane pliki i skopiowane do folderów `data/small/` (5 plików) oraz `data/medium/` (45 plików).*

---

## Krok 3: Uruchomienie wersji sekwencyjnej (Baseline)

Gdy środowisko i dane są gotowe, można przetestować właściwą logikę tworzenia grafu wiedzy pracującą tylko na 1 procesie CPU (Baseline).

Aby przetworzyć mały zestaw danych (wynik zapisze się jako plik grafu `.graphml`):

```powershell
python baseline_sequential.py --data data/small --out results/graph_small.graphml
```

Aby przetworzyć średni zestaw danych:

```powershell
python baseline_sequential.py --data data/medium --out results/graph_medium.graphml
```

---

## Krok 4: Sprawdzenie poprawności wyników (Testy)

Potwierdzenie, że ekstrakcja encji i łączenie relacji funkcjonuje zgodnie z założeniami:

```powershell
pytest tests/test_correctness.py -v
```

*Oczekiwany efekt: 11/11 testów jednostkowych przechodzi pomyślnie (`PASSED`).*

---

## Krok 5: Uruchomienie obciążeniowego Benchmarku

Zmierzenie bazowego czasu wykonania (na potrzeby przyszłych porównań z testami zrównoleglonymi):

```powershell
python benchmarks/bench_sequential.py --runs 3
```

*Oczekiwany efekt: Skrypt wykona po 3 pomiary dla `data/small` oraz `data/medium`, wypisze ładną tabelę wyników w konsoli i zapisze surowe dane w `results/bench_sequential.csv`.*
