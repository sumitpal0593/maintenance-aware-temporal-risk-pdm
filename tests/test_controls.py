import numpy as np

from maestro_pdm.models.controls import fit_logistic_control


def test_logistic_control_selects_from_prespecified_grid():
    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(300, 5))
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] > 0).astype(int)

    X_tune = rng.normal(size=(120, 5))
    y_tune = (X_tune[:, 0] + 0.5 * X_tune[:, 1] > 0).astype(int)

    grid = [0.01, 0.1, 1.0]
    model, table, best_C = fit_logistic_control(
        X_train,
        y_train,
        X_tune,
        y_tune,
        C_grid=grid,
        seed=42,
    )

    assert best_C in grid
    assert set(table["C"]) == set(grid)
    p = model.predict_proba(X_tune)[:, 1]
    assert np.all((p >= 0) & (p <= 1))
