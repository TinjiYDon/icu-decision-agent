"""Import MIMIC-IV 3.1 CSV.gz files into a PostgreSQL Layer0 database.

- Default source: ../../data/mimic-iv-3.1/mimic-iv-3.1 (hosp/ + icu/)
- Default target: postgresql://icu_dev:icu_dev@127.0.0.1:5432/mimic (public schema)

Usage:
  python scripts/import_mimic_31.py --dry-run          # read-only: print DDL + previews
  python scripts/import_mimic_31.py                    # create tables + COPY (gated by F1)

Compliance gate: only run the real import AFTER F1 (MIMIC DUA/CITI) is confirmed.
The script does not download anything; it reads local files only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$")

DEFAULT_SRC = Path(__file__).resolve().parents[2] / "data" / "mimic-iv-3.1" / "mimic-iv-3.1"
DEFAULT_DSN = "postgresql://icu_dev:icu_dev@127.0.0.1:5432/mimic"


def _cell(value: str) -> str:
    return value.strip()


def infer_type(values: list[str]) -> str:
    """bigint / double precision / timestamp / text, from non-empty samples."""
    present = [_cell(v) for v in values if _cell(v) != ""]
    if not present:
        return "text"
    if all(re.fullmatch(r"-?\d+", v) for v in present):
        return "bigint"
    if all(re.fullmatch(r"-?\d+\.\d+", v) for v in present):
        return "double precision"
    if all(TS_RE.match(v) for v in present):
        return "timestamp"
    return "text"


def read_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return [c.strip() for c in next(csv.reader(fh))]


def sample_rows(path: Path, limit: int = 2000) -> list[list[str]]:
    rows: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return rows
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(row)
    return rows


def ddl_for(table: str, columns: list[str], types: list[str]) -> str:
    cols = ",\n    ".join(f'"{c}" {t}' for c, t in zip(columns, types, strict=True))
    return f'CREATE TABLE IF NOT EXISTS public."{table}" (\n    {cols}\n);'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--dry-run", action="store_true", help="print DDL/previews only, no DB write")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    files = sorted([p for sub in ("hosp", "icu") for p in (args.src / sub).glob("*.csv.gz")])
    if not files:
        print(f"no csv.gz found under {args.src}", file=sys.stderr)
        return 1

    plans: list[tuple[Path, list[str], list[str]]] = []
    for path in files:
        cols = read_header(path)
        rows = sample_rows(path, args.limit)
        types = [infer_type([r[i] for r in rows if i < len(r)]) for i in range(len(cols))]
        plans.append((path, cols, types))
        print(f"-- {path.name}: {len(rows)} rows sampled, {len(cols)} cols")
        if args.dry_run:
            print(ddl_for(path.stem.replace(".csv", ""), cols, types))

    if args.dry_run:
        print("dry-run done (no DB write).")
        return 0

    import psycopg  # real import is gated by F1

    with psycopg.connect(args.dsn, autocommit=True) as conn:
        cur = conn.cursor()
        for path, cols, types in plans:
            table = path.stem.replace(".csv", "")
            cur.execute(ddl_for(table, cols, types))
            with cur.copy(f'COPY public."{table}" FROM STDIN WITH (FORMAT csv, HEADER true, NULL \'\')') as copy:
                with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader, None)  # skip header (COPY HEADER true also skips)
                    while True:
                        chunk = fh.read(1024 * 1024)
                        if not chunk:
                            break
                        copy.write(chunk)
            print(f"imported {table}")
        cur.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
