import torch

from maestro_pdm.models.temporal_gru import TemporalRiskGRU


def test_temporal_risk_gru_output_shapes():
    model = TemporalRiskGRU(
        dynamic_dim=5,
        static_dim=3,
        gru_hidden=7,
        gru_layers=1,
        static_hidden=6,
        fusion_hidden=4,
        dropout=0.1,
    )

    X = torch.randn(3, 8, 5)
    lengths = torch.tensor([8, 5, 3])
    static = torch.randn(3, 3)

    logit, h_temporal, fused = model(X, lengths, static)

    assert logit.shape == (3,)
    assert h_temporal.shape == (3, 7)
    assert fused.shape == (3, 4)


def test_model_has_single_scalar_risk_head():
    model = TemporalRiskGRU(
        dynamic_dim=5,
        static_dim=3,
    )
    assert model.risk_head.out_features == 1
