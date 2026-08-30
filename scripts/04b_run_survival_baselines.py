"""
Convenience wrapper for the final Stage 4B reference implementations.

If auton-survival is unavailable in the current interpreter, run scripts
04b1-04b3 here, run 04b4 in the existing Python-3.10 auton environment, then
return to the main environment for 04b5.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_experiment.yaml")
    args = parser.parse_args()

    scripts = [
        "04b1_run_coxph_reference.py",
        "04b2_run_rsf_reference.py",
        "04b3_run_deep_coxph_reference.py",
        "04b4_run_dsm_reference.py",
        "04b5_compile_reference_baselines.py",
    ]

    base = Path(__file__).resolve().parent

    for script in scripts:
        print("\n" + "=" * 78)
        print(f"RUNNING {script}")
        print("=" * 78)
        subprocess.check_call(
            [
                sys.executable,
                str(base / script),
                "--config",
                args.config,
            ]
        )


if __name__ == "__main__":
    main()
