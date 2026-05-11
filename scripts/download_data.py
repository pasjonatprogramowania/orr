"""
Pobiera dataset 'news-articles-corpus' z Kaggle i kopiuje pliki
do data/small/ (pierwsze 5 plików) oraz data/medium/ (pierwsze 50 plików).
"""
import kagglehub
import shutil
import os
from pathlib import Path

# Pobierz dataset
path = kagglehub.dataset_download("sbhatti/news-articles-corpus")
print("Path to dataset files:", path)

# Sprawdź co jest w pobranym katalogu
src = Path(path)
all_txt = []
for ext in ("*.txt", "*.csv"):
    all_txt.extend(sorted(src.rglob(ext)))

print(f"Znaleziono {len(all_txt)} plików ({', '.join(set(f.suffix for f in all_txt))})")

# Jeśli CSV – użyjemy go inaczej; jeśli TXT – kopiujemy bezpośrednio
txt_files = sorted(src.rglob("*.txt"))
csv_files = sorted(src.rglob("*.csv"))

project_root = Path(__file__).parent

if txt_files:
    small_dst  = project_root / "data" / "small"
    medium_dst = project_root / "data" / "medium"
    small_dst.mkdir(parents=True, exist_ok=True)
    medium_dst.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(txt_files[:50]):
        dst_dir = small_dst if i < 5 else medium_dst
        shutil.copy(f, dst_dir / f.name)
    print(f"Skopiowano {min(5, len(txt_files))} plików → data/small/")
    print(f"Skopiowano {max(0, min(50, len(txt_files)) - 5)} plików → data/medium/")

elif csv_files:
    # Dataset jest w CSV – podzielimy go na małe pliki TXT
    import pandas as pd
    df = pd.read_csv(csv_files[0])
    print(f"CSV columns: {df.columns.tolist()}")
    print(df.head(2))

    # Znajdź kolumnę z tekstem
    text_col = None
    for candidate in ["content", "text", "article", "body", "description"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        text_col = df.columns[-1]  # ostatnia kolumna jako fallback
    print(f"Używam kolumny: '{text_col}'")

    df = df.dropna(subset=[text_col])

    small_dst  = project_root / "data" / "small"
    medium_dst = project_root / "data" / "medium"
    small_dst.mkdir(parents=True, exist_ok=True)
    medium_dst.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= 200:
            break
        dst_dir = small_dst if i < 10 else medium_dst
        fname = f"article_{i:04d}.txt"
        (dst_dir / fname).write_text(str(row[text_col]), encoding="utf-8")

    print(f"Zapisano 10 plików → data/small/")
    print(f"Zapisano {min(190, len(df)-10)} plików → data/medium/")

print("Gotowe!")
