"""stdio MCP server — tool predict_risk.

Install optional extra then run:

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\pip.exe install "mcp>=1.0"
.\.venv\Scripts\python.exe -m presentation.mcp_server
```
"""

from __future__ import annotations

from presentation.mcp_tools import PREDICT_RISK_SCHEMA, predict_risk


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing optional dependency `mcp`. Install with: pip install 'mcp>=1.0'"
        ) from exc

    server = FastMCP("icu-decision-predict")

    @server.tool(
        name=PREDICT_RISK_SCHEMA["name"],
        description=PREDICT_RISK_SCHEMA["description"],
    )
    def predict_risk_tool(stay_id: int) -> dict:
        return predict_risk(stay_id)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
