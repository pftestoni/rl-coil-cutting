# Usage: Run from repo root. Generates patterns/patterns_60.csv using grande (60 SKUs)
# Adjust paths if needed.

python -m tools.generate_patterns --catalog catalogs/catalog_60.json --env grande --out patterns/patterns_60.csv --trim-max 40 --max-patterns 1200 --seed 123
