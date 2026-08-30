from pathlib import Path
import platform
import sys

import pandas as pd
import torch

from maestro_pdm.config import get_data_dir, load_config
from maestro_pdm.data.loading import validate_dataset_directory


def main() -> None:
    cfg = load_config("configs/paper_experiment.yaml")
    data_dir = get_data_dir()
    validate_dataset_directory(data_dir)

    print("=" * 70)
    print("Maintenance-Aware PdM: environment check")
    print("=" * 70)
    print("Python :", sys.version.split()[0])
    print("OS     :", platform.platform())
    print("PyTorch:", torch.__version__)
    print("CUDA   :", torch.cuda.is_available())
    print("Data   :", data_dir)
    print("tau    :", cfg["experiment"]["tau"])
    print()

    labels = pd.read_csv(data_dir / "validation_labels.csv")
    print("validation_labels.csv")
    print("  rows   :", len(labels))
    print("  columns:", list(labels.columns))
    print()
    print("Setup check completed successfully.")


if __name__ == "__main__":
    main()
