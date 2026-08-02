import numpy as np

from application.demo_curves import net_benefit_curve


def test_net_benefit_curve_shape():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(int)
    p = rng.random(200)
    out = net_benefit_curve(y, p, thresholds=np.array([0.1, 0.3, 0.5]))
    assert out["status"] == "ok"
    assert len(out["thresholds"]) == 3
    assert len(out["net_benefit_model"]) == 3
    assert len(out["net_benefit_treat_all"]) == 3
