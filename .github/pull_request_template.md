## 人读摘要

| 项 | 内容 |
|----|------|
| Owner | A / B / C |
| 关联 | Closes # |

## 做了什么

## 如何验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

## 给 AI 的上下文

```text
分层：UI 只调 L4；见 docs/adr/001-layer-boundaries.md
参数：docs/PARAM_STORY.md
禁提交：dumps/ artifacts/
```
