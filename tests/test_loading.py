import pandas as pd

from maestro_pdm.data.loading import detect_id_time_columns


def test_detect_id_time_columns():
    df = pd.DataFrame(
        {
            "vehicle_id": [1, 1],
            "time_step": [0, 1],
            "sensor_1": [0.1, 0.2],
        }
    )
    assert detect_id_time_columns(df) == ("vehicle_id", "time_step")
