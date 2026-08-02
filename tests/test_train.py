def test_train_module_imports():
    from domain.features.build import build_features
    from domain.labels.mortality_12h import build_labels
    from domain.models.lgbm import train_and_save

    assert callable(build_features)
    assert callable(build_labels)
    assert callable(train_and_save)


def test_train_can_reuse_existing_feature_and_label_tables(monkeypatch):
    import application.train as train_module

    monkeypatch.setattr(
        train_module,
        "build_features",
        lambda: (_ for _ in ()).throw(AssertionError("must not rebuild features")),
    )
    monkeypatch.setattr(
        train_module,
        "build_labels",
        lambda: (_ for _ in ()).throw(AssertionError("must not rebuild labels")),
    )
    monkeypatch.setattr(
        train_module,
        "train_and_save",
        lambda: {
            "train_n": 70,
            "val_n": 10,
            "test_n": 20,
            "auc_val": 0.7,
            "auc_test": 0.68,
            "pr_auc_val": 0.1,
            "pr_auc_test": 0.09,
            "brier_val": 0.1,
            "brier_test": 0.1,
            "pos_rate": 0.02,
            "positive": 2,
            "stratified": True,
        },
    )

    result = train_module.run_train(rebuild_data=False)

    assert result["data_mode"] == "existing_feat_label"
    assert result["status"] == "train_ok"
