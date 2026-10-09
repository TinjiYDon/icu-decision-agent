"""CLI: TFT-lite synthetic smoke → artifacts/d3/tft_lite.json"""

from __future__ import annotations

import json

from domain.models.temporal.tft_smoke import run_smoke


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
