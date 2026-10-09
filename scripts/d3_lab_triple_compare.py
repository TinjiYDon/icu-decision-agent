"""CLI: lab-only LGBM vs GRU-D vs TFT-lite deepen (H2)."""

from __future__ import annotations

import argparse
import json

from domain.models.temporal.lab_temporal import run_lab_triple_compare


def main() -> None:
    p = argparse.ArgumentParser(description="Lab temporal triple compare")
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--lookback", type=int, default=12)
    p.add_argument("--epochs", type=int, default=25)
    args = p.parse_args()
    payload = run_lab_triple_compare(
        limit=args.limit, lookback_hours=args.lookback, epochs=args.epochs
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
