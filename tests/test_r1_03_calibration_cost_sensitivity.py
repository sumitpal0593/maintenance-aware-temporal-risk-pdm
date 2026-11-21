"""Unit tests for R1-03 additive sensitivity script."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'r1_03_calibration_cost_sensitivity.py'
spec = spec_from_file_location('r1_03_calibration_cost_sensitivity', PATH)
m = module_from_spec(spec)
spec.loader.exec_module(m)


def test_odds_shift_identity_and_direction():
    p = np.array([0.02, 0.1, 0.5, 0.8])
    assert np.allclose(m.odds_shift(p, .1, .1), p)
    shifted = m.odds_shift(p, .1, .02)
    assert np.all(shifted < p)
    assert np.all(np.diff(shifted) > 0)


def test_cost_threshold_and_benefit():
    p = np.array([.1, .15104166666666666, .18045722])
    benefits, actions = m.benefit_and_action(p, ca=145, cf=1200, rho=.2)
    assert np.isclose(benefits[1], 0)
    assert actions.tolist() == [False, False, True]
    assert np.isclose(benefits[2], 28.2389312, atol=1e-6)


def test_input_duplicate_vehicle_rejected(tmp_path):
    path = tmp_path / 'dup.csv'
    pd.DataFrame({'vehicle_id':[1,1], 'binary_label':[0,1], 'ipcw_platt_probability':[0.1,.2]}).to_csv(path,index=False)
    try:
        m.load_predictions(path)
        assert False, 'Expected duplicate-ID failure'
    except ValueError as exc:
        assert 'exactly one' in str(exc)


def test_full_analysis_outputs_on_toy_data(tmp_path):
    inp = tmp_path / 'toy.csv'
    pd.DataFrame({'vehicle_id':np.arange(1,41),'binary_label':[1]*4+[0]*36,'ipcw_platt_probability':np.linspace(.03,.4,40)}).to_csv(inp,index=False)
    cfg = tmp_path / 'cost.yaml'
    cfg.write_text('scenarios:\n  high_consequence:\n    C_A: 145\n    C_F: 1200\n    rho: 0.2\n')
    result = m.run_analysis(inp,cfg,tmp_path/'out')
    assert result['n_validation'] == 40
    assert (tmp_path/'out'/'r1_03_scenario_policy_sensitivity.csv').is_file()
    assert (tmp_path/'out'/'r1_03_high_consequence_p90_benefit.png').is_file()
    risks = pd.read_csv(tmp_path/'out'/'r1_03_vehicle_risks_actions.csv')
    assert len(risks) == 40
