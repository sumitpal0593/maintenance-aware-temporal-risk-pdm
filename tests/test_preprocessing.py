from pathlib import Path

import numpy as np
import pandas as pd

from maestro_pdm.data.preprocessing import (
    OperationalPreprocessor,
    fit_static_encoder,
    static_encoding_summary,
    transform_static_specs,
)


def test_operational_transform_mean_imputes_to_zero():
    p = OperationalPreprocessor(
        feature_names=["a", "b"],
        means=np.array([10.0, 20.0]),
        stds=np.array([2.0, 5.0]),
        observed_counts=np.array([5, 5]),
        total_rows=5,
    )

    X = np.array([[12.0, np.nan], [8.0, 25.0]])
    Z = p.transform(X)

    np.testing.assert_allclose(
        Z,
        np.array([[1.0, 0.0], [-1.0, 1.0]], dtype=np.float32),
    )


def test_static_encoder_is_training_only_and_ignores_unknown():
    specs = pd.DataFrame(
        {
            "vehicle_id": [1, 2, 3],
            "Spec_0": ["Cat0", "Cat1", "Cat2"],
            "Spec_1": ["A", "A", "B"],
        }
    )

    encoder, id_col, feature_cols = fit_static_encoder(
        specs,
        allowed_vehicle_ids={1, 2},
    )
    overview, _ = static_encoding_summary(encoder, feature_cols)
    vectors = transform_static_specs(
        encoder,
        specs,
        id_col,
        feature_cols,
    )

    assert int(overview.loc[0, "original_static_dimension_q"]) == 2
    # Cat0/Cat1 + A = 3 fitted columns; Cat2/B are unseen and ignored.
    assert int(overview.loc[0, "encoded_static_dimension_q_e"]) == 3
    assert vectors[3].shape == (3,)
