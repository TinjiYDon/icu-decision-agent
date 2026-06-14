def test_train_module_imports():
    from domain.features.build import build_features
    from domain.labels.mortality_12h import build_labels
    from domain.models.lgbm import train_and_save

    assert callable(build_features)
    assert callable(build_labels)
    assert callable(train_and_save)
