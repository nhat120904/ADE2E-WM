"""Check that two NAVSIM score CSVs hold the same tokens and scores (exit 1 otherwise)."""

import sys

import numpy as np
import pandas as pd


def main(ref_path, new_path, atol=1e-9):
    ref, new = (pd.read_csv(p, index_col=0).set_index("token").sort_index() for p in (ref_path, new_path))
    problems = []
    if list(ref.index) != list(new.index):
        problems.append(f"token sets differ: {len(ref)} vs {len(new)} rows, {len(set(ref.index) ^ set(new.index))} not shared")
    if list(ref.columns) != list(new.columns):
        problems.append(f"columns differ: {set(ref.columns) ^ set(new.columns)}")
    if not problems:
        for col in ref.columns:
            a, b = ref[col], new[col]
            if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
                same = np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=atol, equal_nan=True)
            else:
                same = (a.astype(str) == b.astype(str)).to_numpy()
            if not same.all():
                problems.append(f"{col}: {(~same).sum()} rows differ")
    print(f"ref={ref_path}\nnew={new_path}\nrows={len(ref)} cols={len(ref.columns)}")
    print("\n".join(problems) if problems else "IDENTICAL (atol=%g)" % atol)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
