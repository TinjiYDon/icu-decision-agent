def test_predict_patient_imports():
    from application.predict_patient import list_stays, predict_patient

    assert callable(list_stays)
    assert callable(predict_patient)


def test_predict_patient_no_model():
    from application.predict_patient import predict_patient

    out = predict_patient(999999999)
    assert "stay_id" in out
    assert out["status"] in ("ok", "no_model", "no_features")


def test_recommend_action_bands():
    from domain.models.lgbm import recommend_action

    assert recommend_action(0.1)["band"] == "observe"
    assert recommend_action(0.3)["band"] == "recheck"
    assert recommend_action(0.5)["band"] == "monitor"
    assert recommend_action(0.8)["band"] == "escalate"
    assert recommend_action(0.5, score_kind="raw")["band"] == "unknown"
