# 队友：恢复 Layer1 dump

1. 安装 Docker Desktop，`docker compose up -d`（compose 文件在团队 `project-code/` 或向负责人索取）
2. 网盘下载 `icu_decision_layer1_schemas_*.dump`
3. `.\restore_layer1.ps1 -Target decision -DumpFile .\dumps\xxx.dump`
4. 配置 `.env` 与 `configs/database.yaml`
5. `pytest` + `streamlit run presentation/streamlit_app.py`

**不要**将 dump 提交 GitHub。
