# Copilot / Agent instructions — icu-decision-agent

- L4 only from UI: `application.predict_patient`
- Test: `pytest tests/test_predict.py -q` with `PYTHONPATH=.`
- Labels/thresholds: `configs/labels.yaml` → `recommend` bands
- Feature leak risk: `hospital_expire_flag`
- No PPO/RL in this repo; supervised + SHAP
- Do not commit dumps/artifacts
- Read `docs/PARAM_STORY.md` before changing features/labels
