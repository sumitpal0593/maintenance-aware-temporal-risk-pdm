from maestro_pdm.config import load_config


def test_paper_horizon_is_48():
    cfg = load_config("configs/paper_experiment.yaml")
    assert cfg["experiment"]["tau"] == 48


def test_validation_is_not_artificially_limited():
    cfg = load_config("configs/paper_experiment.yaml")
    assert cfg["data"]["max_validation_vehicles"] is None
