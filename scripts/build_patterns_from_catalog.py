import argparse
import csv
import json
from pathlib import Path


def load_catalog(path: Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Supports list of {sku, len_mm}
    lens = [int(item["len_mm"]) for item in data]
    return lens


def gen_patterns(
    lengths_mm,
    coil_width_mm: int,
    kerf_mm: int,
    trim_max_mm: int,
    max_patterns: int = 2000,
    max_per_type: int = 2,
    max_per_pair: int = 2,
):
    n = len(lengths_mm)
    patterns = []
    seen = set()

    def add_pattern(counts):
        key = tuple(counts)
        if key in seen:
            return False
        # physics check: used + kerf <= coil, leftover within trim
        pieces = sum(counts)
        used = sum(c * L for c, L in zip(counts, lengths_mm)) + max(0, pieces - 1) * kerf_mm
        if used > coil_width_mm:
            return False
        scrap = coil_width_mm - used
        if scrap < 0 or scrap > trim_max_mm:
            return False
        seen.add(key)
        patterns.append(counts[:])
        return True

    # Singles: for each type, try best fill and one alternate
    for i, L in enumerate(lengths_mm):
        best = coil_width_mm // max(1, L)
        added = 0
        k = best
        while k > 0 and added < max_per_type and len(patterns) < max_patterns:
            counts = [0] * n
            counts[i] = k
            if add_pattern(counts):
                added += 1
            k -= 1
        if len(patterns) >= max_patterns:
            break

    # Pairs: greedy fill per pair
    for i, Li in enumerate(lengths_mm):
        for j, Lj in enumerate(lengths_mm):
            if len(patterns) >= max_patterns:
                break
            added = 0
            # try a few k_i values around best
            best_i = coil_width_mm // max(1, Li)
            for ki in range(best_i, max(0, best_i - 5), -1):
                if ki <= 0 or added >= max_per_pair or len(patterns) >= max_patterns:
                    break
                pieces_ki = ki
                used_i = ki * Li + max(0, pieces_ki - 1) * kerf_mm
                if used_i > coil_width_mm:
                    continue
                rem = coil_width_mm - used_i
                # account for kerf if we add any j pieces
                # approximate: assume at least one extra kerf if ki>0 and kj>0
                max_kj = (rem - kerf_mm) // Lj if rem > kerf_mm else 0
                if max_kj <= 0:
                    continue
                # search kj downwards to hit trim window
                for kj in range(int(max_kj), max(0, int(max_kj) - 5), -1):
                    pieces = ki + kj
                    used = ki * Li + kj * Lj + max(0, pieces - 1) * kerf_mm
                    if used > coil_width_mm:
                        continue
                    scrap = coil_width_mm - used
                    if 0 <= scrap <= trim_max_mm:
                        counts = [0] * n
                        counts[i] = ki
                        counts[j] = kj
                        if add_pattern(counts):
                            added += 1
                            break
                if added >= max_per_pair or len(patterns) >= max_patterns:
                    break
        if len(patterns) >= max_patterns:
            break

    return patterns


def write_patterns_csv(out_path: Path, lengths_mm, patterns):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [str(L) for L in lengths_mm]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in patterns:
            w.writerow([int(x) for x in row])


def main():
    ap = argparse.ArgumentParser(description="Generate patterns CSV from catalog (mm-columns)")
    ap.add_argument("--catalog", required=True, help="Path to catalogs/catalog_XX.json")
    ap.add_argument("--out", required=True, help="Output CSV path (patterns/patterns_XX.csv)")
    ap.add_argument("--coil-width", type=int, default=1200)
    ap.add_argument("--kerf", type=int, default=0)
    ap.add_argument("--trim-max", type=int, default=40)
    ap.add_argument("--max-patterns", type=int, default=2000)
    args = ap.parse_args()

    lengths = load_catalog(Path(args.catalog))
    pats = gen_patterns(
        lengths,
        coil_width_mm=int(args.coil_width),
        kerf_mm=int(args.kerf),
        trim_max_mm=int(args.trim_max),
        max_patterns=int(args.max_patterns),
    )
    write_patterns_csv(Path(args.out), lengths, pats)
    print(f"[BUILD] Wrote {len(pats)} patterns to {args.out}")


if __name__ == "__main__":
    main()
