from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_axes():
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    return fig, ax


def save_reliability_diagram(
    reliability_table: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = _prepare_axes()

    for method, group in reliability_table.groupby("method", sort=False):
        ax.plot(
            group["mean_probability"],
            group["observed_event_rate"],
            marker="o",
            linewidth=1.5,
            label=method,
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Validation reliability diagram")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_probability_histogram(
    probabilities,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    p = np.asarray(probabilities, dtype=float).reshape(-1)

    fig, ax = _prepare_axes()
    ax.hist(p, bins=40)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Vehicle count")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
