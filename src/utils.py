import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_patterns_csv(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    # Keep only numeric columns (counts). Assumes rows = patterns, cols = SKUs.
    counts = df.select_dtypes(include=["number"]).to_numpy(dtype=np.int32)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("CSV de padrões inválido (linhas=padrões, colunas=SKUs)")
    return counts

@dataclass
class Catalog:
    sku_ids: list
    widths_mm: np.ndarray  # shape (N,)

    @staticmethod
    def from_json(path: str, expect_n: int | None = None) -> "Catalog":
        data = load_json(path)
        sku_ids, widths = [], []
        for row in data:
            sku_ids.append(row.get("sku"))
            widths.append(row.get("len_mm"))
        widths = np.asarray(widths, dtype=np.int32)
        if expect_n is not None and len(sku_ids) != expect_n:
            raise ValueError(f"Catálogo tem {len(sku_ids)} SKUs, esperado {expect_n}")
        return Catalog(sku_ids=sku_ids, widths_mm=widths)

def ensure_shapes(patterns: np.ndarray, catalog: Catalog) -> np.ndarray:
    if patterns.shape[1] != catalog.widths_mm.shape[0]:
        raise ValueError(
            f"Padrões com {patterns.shape[1]} colunas, catálogo tem {catalog.widths_mm.shape[0]} SKUs"
        )
    return patterns
