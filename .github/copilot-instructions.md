# Copilot / Agent instructions — icu-decision-agent

- L4 only from UI: `application.predict_patient`
- Test: `pytest tests/test_features_leak.py tests/test_predict.py -q` with `PYTHONPATH=.`
- Labels/thresholds: `configs/labels.yaml` → `recommend` bands
- Features: `configs/features.yaml` — DENIED `hospital_expire_flag`, total `los_hours`
- Split: train/val/test 0.7/0.1/0.2 by stay_id seed=42 (`domain/models/split.py`)
- Wave2: B retrain and write auc_val/auc_test to STATUS
- No PPO/RL; supervised + SHAP
- Do not commit dumps/artifacts
- Read `docs/PARAM_STORY.md` and `docs/ROADMAP_EXEC.md`
