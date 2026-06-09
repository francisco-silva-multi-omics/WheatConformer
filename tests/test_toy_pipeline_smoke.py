from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_requested_outputs import compute_hmp_qc
from server_training_pipeline.trait_isolation import select_single_trait


ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_SCRIPT = ROOT / "build_gaussian_genomic_kernel.py"


def test_toy_preprocessing_to_model_inputs_smoke(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "m1": [0, 0, 2, -9],
            "m2": [0, 2, 2, 2],
            "m3": [2, 1, 0, 1],
            "m4": [0, -9, 2, 2],
        }
    )
    thresholds = {
        "maf_min": 0.0,
        "marker_het_max": 1.0,
        "sample_het_max": 1.0,
        "marker_missing_max": 1.0,
        "sample_missing_max": 1.0,
    }
    qc = compute_hmp_qc(raw, pd.Series(["g1", "g2", "g3", "g4"]), thresholds)
    linear = qc["kernel"]
    assert linear.shape == (4, 4)
    assert np.isfinite(linear).all()

    linear_path = tmp_path / "K_G.npy"
    order_path = tmp_path / "K_G_order.tsv"
    gaussian_path = tmp_path / "K_G_RBF.npy"
    gaussian_qc_path = tmp_path / "K_G_RBF.qc.json"
    np.save(linear_path, linear)
    pd.DataFrame({"sample_id": qc["sample_ids"]}).to_csv(order_path, sep="\t", index=False)

    subprocess.run(
        [
            sys.executable,
            str(GAUSSIAN_SCRIPT),
            "--linear-kernel",
            str(linear_path),
            "--sample-order",
            str(order_path),
            "--out-kernel",
            str(gaussian_path),
            "--out-qc",
            str(gaussian_qc_path),
            "--gamma-multiplier",
            "1.5",
            "--median-sample-size",
            "4",
            "--psd-sample-size",
            "4",
            "--chunk-size",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    gaussian = np.load(gaussian_path)
    gaussian_qc = json.loads(gaussian_qc_path.read_text(encoding="utf-8"))
    assert gaussian.shape == linear.shape
    assert np.allclose(np.diag(gaussian), 1.0)
    assert gaussian_qc["number_of_samples"] == 4

    observations = pd.DataFrame(
        {
            "trait_name_canonical": ["Grain Yield"] * 4,
            "genotype_id": qc["sample_ids"],
            "env_id": ["e1", "e1", "e2", "e2"],
            "response": [1.0, 2.0, 1.5, 2.5],
        }
    )
    selected, trait = select_single_trait(observations, None)
    assert trait == "Grain Yield"
    assert set(selected["genotype_id"]) == set(qc["sample_ids"])
