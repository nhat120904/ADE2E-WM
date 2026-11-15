"""Phase 0 audit of kesai-labs/nuplan (123D format, 10 Hz) arrow files.

Prints the schema of every arrow file of the downloaded logs, row counts, timestamp spacing,
and, where the columns exist, box-track persistence and per-camera timestamp offsets.
Column names are discovered from the schema; nothing is assumed beyond what is printed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def read_arrow(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as src:
        try:
            return ipc.open_file(src).read_all()
        except pa.ArrowInvalid:
            src.seek(0)
            return ipc.open_stream(src).read_all()


def short(v, n=160):
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


def ts_column(table: pa.Table):
    for name in table.column_names:
        if "timestamp" in name.lower() or name.lower() in ("time_us", "time"):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {}
    for log_dir in sorted((args.root / "logs").glob("*/*")):
        rep = {}
        for f in sorted(log_dir.glob("*.arrow")):
            if f.stat().st_size == 0:
                rep[f.name] = "EMPTY/MISSING"
                continue
            t = read_arrow(f)
            # do not materialise image bytes
            info = {"rows": t.num_rows, "schema": [f"{fld.name}: {short(fld.type, 120)}" for fld in t.schema]}
            meta = t.schema.metadata or {}
            info["schema_metadata_keys"] = [k.decode() for k in meta]
            info["schema_metadata_sample"] = {k.decode(): short(v.decode(errors="replace"), 400) for k, v in meta.items()}
            first = {}
            for name in t.column_names:
                col = t.column(name)
                if pa.types.is_binary(col.type) or pa.types.is_large_binary(col.type):
                    first[name] = f"<binary {len(col[0].as_py() or b'')} bytes>"
                else:
                    first[name] = short(col[0].as_py() if t.num_rows else None)
            info["first_row"] = first
            tc = ts_column(t)
            if tc and t.num_rows > 1:
                ts = np.array(t.column(tc).to_pylist(), dtype=np.float64)
                d = np.diff(ts)
                info["timestamp_column"] = tc
                info["dt_median"] = float(np.median(d))
                info["dt_p5_p95"] = [float(np.percentile(d, 5)), float(np.percentile(d, 95))]
                info["duration"] = float(ts[-1] - ts[0])
            rep[f.name] = info
        report[log_dir.name] = rep
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:30000])


if __name__ == "__main__":
    main()
