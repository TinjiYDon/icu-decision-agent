def test_predict_patient_imports():
    from application.predict_patient import list_stays, predict_patient

    assert callable(list_stays)
    assert callable(predict_patient)


def test_predict_patient_no_model():
    from application.predict_patient import predict_patient

    out = predict_patient(999999999)
    assert "stay_id" in out
    assert out["status"] in ("ok", "no_model", "no_features")
