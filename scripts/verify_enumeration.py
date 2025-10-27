import json, csv, sys
from pathlib import Path
from typing import List, Tuple

def load_lengths(catalog_path: Path, n: int=None) -> List[int]:
    with open(catalog_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out: List[int] = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict):
                out.append(int(it.get('len_mm', it.get('length', it.get('mm')))))
            else:
                out.append(int(it))
    elif isinstance(data, dict):
        # Assume mapping sku->len
        for k in sorted(data.keys()):
            out.append(int(data[k]))
    else:
        raise ValueError('Unsupported catalog format')
    if n is not None:
        out = out[:n]
    return out

def enumerate_all(lengths: List[int], coil: int, trim_max: int):
    # DFS over counts in DESC length order; accept only at leaf
    # Ensure original order mapping at the end
    import numpy as np
    arr = np.array(lengths, dtype=int)
    order = np.argsort(-arr)
    arr_d = arr[order]
    ub = coil // arr_d
    n = len(arr_d)
    # suffix capacity bound
    suffix_cap = [0]*(n+1)
    for i in range(n-1, -1, -1):
        suffix_cap[i] = suffix_cap[i+1] + int(arr_d[i])*int(ub[i])
    min_target = max(0, coil - trim_max)
    counts = [0]*n
    inv = list(np.argsort(order))
    res = set()
    def emit():
        v = [0]*n
        for i in range(n):
            v[i] = counts[inv[i]]  # map back
        return tuple(int(x) for x in v)
    def dfs(i: int, s: int, p: int):
        if s > coil:
            return
        if s + suffix_cap[i] < min_target:
            return
        if i == n:
            if min_target <= s <= coil:
                res.add(emit())
            return
        L = int(arr_d[i])
        for c in range(int(ub[i]), -1, -1):
            counts[i] = c
            dfs(i+1, s + c*L, p + c)
        counts[i] = 0
    dfs(0, 0, 0)
    return res

def read_generated(csv_path: Path) -> List[Tuple[int,...]]:
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r)
        out = []
        for row in r:
            out.append(tuple(int(x) for x in row))
        return out

if __name__ == '__main__':
    cat = Path(sys.argv[1] if len(sys.argv) > 1 else 'catalogs/catalog_20.json')
    csvp = Path(sys.argv[2] if len(sys.argv) > 2 else 'patterns/patterns_20_trim399_all.csv')
    coil = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    trim_max = int(sys.argv[4]) if len(sys.argv) > 4 else 399
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    lens = load_lengths(cat)
    lens = lens[:n]
    print(f'[INFO] lengths({n}):', lens)
    got = set(read_generated(csvp))
    print('[INFO] generated count:', len(got))
    allp = enumerate_all(lens, coil, trim_max)
    print('[INFO] enumerated count:', len(allp))
    missing = allp - got
    extra = got - allp
    print('[CHECK] missing in CSV:', len(missing))
    print('[CHECK] extra in CSV:', len(extra))
    if missing:
        print('> sample missing:', next(iter(missing)))
    if extra:
        print('> sample extra:', next(iter(extra)))
