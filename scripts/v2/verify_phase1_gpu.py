"""Fail-closed TensorFlow GPU verification for the Phase 1 environment."""

from __future__ import annotations

import json
import platform

import tensorflow as tf


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("TensorFlow did not enumerate a GPU")

    with tf.device("/GPU:0"):
        value = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))

    report = {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "cuda_build": tf.test.is_built_with_cuda(),
        "build_info": dict(tf.sysconfig.get_build_info()),
        "gpus": [gpu.name for gpu in gpus],
        "smoke_sum": float(tf.reduce_sum(value).numpy()),
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if report["smoke_sum"] != 32768.0:
        raise SystemExit(f"Unexpected GPU smoke-test sum: {report['smoke_sum']}")


if __name__ == "__main__":
    main()
