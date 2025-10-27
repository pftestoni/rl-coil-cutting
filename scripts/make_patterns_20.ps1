# Usage: Run from repo root. Generates patterns/patterns_20.csv using medio (20 SKUs)
# Adjust paths if needed.

python -m tools.generate_patterns --catalog catalogs/catalog_60.json --env medio --out patterns/patterns_20.csv --trim-max 40 --max-patterns 1200 --seed 123
