# Instrukcja uruchomienia projektu (KG Pipeline)

Projekt można uruchomić na dwa sposoby: błyskawicznie poprzez Docker (wraz z graficznym interfejsem webowym) lub lokalnie ze środowiskiem Python.

## Opcja A: Uruchomienie przez Docker (Zalecane)

Dzięki konteneryzacji nie musisz instalować lokalnie Pythona, paczek ani modelu spaCy.

1. **Uruchomienie kontenerów:**
   Będąc w głównym katalogu projektu (`ORR`), wpisz:

   ```bash
   docker-compose up --build
   ```
2. **Dostęp do UI:**
   Aplikacja automatycznie wystawi interfejs pod adresem:
    **http://localhost:8765**

   W UI możesz interaktywnie zmieniać wielkość zbioru danych (`small`/`medium`), tryby przetwarzania (`sequential`/`parallel`/`distributed`) oraz podglądać wynikowy graf.

---

## Opcja B: Uruchomienie lokalne (CLI)

Wymagany Python 3.10+ (zalecany 3.12).

### 1. Przygotowanie środowiska i danych

Otwórz terminal w katalogu projektu (`ORR`):

```powershell
# Utworzenie i aktywacja wirtualnego środowiska
python -m venv venv
.\venv\Scripts\activate

# Instalacja zależności
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Pobranie danych testowych
python scripts/download_data.py
```

### 2. Uruchamianie benchmarków

Skrypty znajdują się w folderze `benchmarks/`.

**Wersja Sekwencyjna (Baseline):**

```powershell
python benchmarks/bench_sequential.py --runs 3
```

**Wersja Równoległa (Shared Memory):**

```powershell
# Szybki test
python benchmarks/parallel_benchmark.py --data data/small

# Pełny test
python benchmarks/parallel_benchmark.py --data data/medium
```

**Wersja Rozproszona (Master-Worker IPC):**

```powershell
python benchmarks/distributed_benchmark.py --data data/medium --workers 4 --batch-size 2
```

### 3. Testy poprawności

Aby upewnić się, że cała logika (włączając protokoły kolejek) działa poprawnie:

```powershell
pytest tests/ -v --tb=short
```

### 4. Lokalny Serwer UI

Jeśli chcesz odpalić UI poza Dockerem:

```powershell
python ui/ui_server.py
# Następnie wejdź na http://localhost:8765
```
