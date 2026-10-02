import os
import csv
from typing import List, Dict, Any

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
