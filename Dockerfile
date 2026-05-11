FROM python:3.12-slim

# Instalacja podstawowych narzędzi i języka polskiego/angielskiego
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalacja zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pobranie modelu językowego
RUN python -m spacy download en_core_web_sm

# Kopiowanie plików projektu
COPY . .

# Wystawienie portu dla UI
EXPOSE 8765

# Domyślne polecenie (uruchomienie UI servera)
CMD ["python", "ui_server.py"]
