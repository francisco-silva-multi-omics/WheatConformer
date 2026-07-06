from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


def make_one_hot(seq: np.ndarray) -> np.ndarray:
    x = np.zeros((len(seq), 4), dtype=np.float32)
    ok = seq < 4
    x[np.where(ok)[0], seq[ok]] = 1.0
    return x


class H5WindowSequence(tf.keras.utils.Sequence):
    def __init__(self, h5_path: Path, indices: np.ndarray, batch_size: int, shuffle: bool = False):
        super().__init__()
        self.h5_path = h5_path
        self.indices = indices.astype(np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.h5 = None
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def _open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, batch_idx: int):
        self._open()
        idx = self.indices[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
        order = np.argsort(idx)
        inverse = np.argsort(order)
        sorted_idx = idx[order]
        seq = self.h5["seq"][sorted_idx].astype(np.int64)[inverse]
        y = self.h5["signal"][sorted_idx].astype(np.float32)[inverse]
        x = np.stack([make_one_hot(s) for s in seq], axis=0)
        y = np.transpose(y, (0, 2, 1))
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


@tf.keras.utils.register_keras_serializable(package="WheatRegulatory")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=max(channels // heads, 1), dropout=dropout)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(channels * 4, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(channels),
            ]
        )
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, training=training), training=training)
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h, training=training), training=training)
        return x


def build_enformer_lite(window_size: int, bins: int, tracks: int, channels: int, layers: int, heads: int, dropout: float):
    if window_size % bins != 0:
        raise ValueError(f"window_size {window_size} must be divisible by bins {bins}")
    pool_size = window_size // bins
    seq_in = tf.keras.Input(shape=(window_size, 4), name="sequence_onehot")
    x = tf.keras.layers.Conv1D(channels, 15, padding="same")(seq_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Conv1D(channels, 7, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=pool_size, padding="valid")(x)
    for _ in range(layers):
        x = TransformerBlock(channels, heads, dropout)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    out = tf.keras.layers.Dense(tracks, activation="softplus", name="multiomics_signal")(x)
    return tf.keras.Model(seq_in, out, name="wheat_enformer_lite_tf")


def random_split_indices(n: int, seed: int, val_fraction: float, test_fraction: float):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def grouped_split_indices(groups: np.ndarray, seed: int, val_fraction: float, test_fraction: float):
    groups = groups.astype(str)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise SystemExit(f"Grouped regulatory split requires at least 3 groups; found {len(unique)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    counts = {group: int(np.sum(groups == group)) for group in unique}
    targets = {"test": len(groups) * test_fraction, "val": len(groups) * val_fraction}
    assigned: dict[str, set[str]] = {"test": set(), "val": set()}
    remaining = list(unique)
    for split in ("test", "val"):
        total = 0
        while remaining and (total < targets[split] or not assigned[split]):
            group = remaining.pop(0)
            assigned[split].add(group)
            total += counts[group]
    test = np.where(np.isin(groups, list(assigned["test"])))[0]
    val = np.where(np.isin(groups, list(assigned["val"])))[0]
    train = np.where(~np.isin(groups, list(assigned["test"] | assigned["val"])))[0]
    if min(len(train), len(val), len(test)) == 0:
        raise SystemExit(f"Empty grouped split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("regulatory_model/enformer_like_tf"))
    parser.add_argument("--prefix", default="wheat_enformer_lite_tf")
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split",
        choices=["chromosome", "subgenome", "random"],
        default="chromosome",
        help="Use chromosome holdouts by default to prevent leakage between nearby regulatory windows.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print(f"TensorFlow: {tf.__version__}; GPUs: {len(tf.config.list_physical_devices('GPU'))}", flush=True)

    with h5py.File(args.h5, "r") as h5:
        n = h5["seq"].shape[0]
        window_size = h5["seq"].shape[1]
        tracks = h5["signal"].shape[1]
        bins = h5["signal"].shape[2]
        chrom = np.asarray(h5["chrom"]).astype("U")
        subgenome = np.asarray(h5["subgenome"]).astype("U") if "subgenome" in h5 else np.full(n, "unknown")
    if args.split == "random":
        train_idx, val_idx, test_idx = random_split_indices(n, args.seed, args.val_fraction, args.test_fraction)
    else:
        groups = chrom if args.split == "chromosome" else subgenome
        train_idx, val_idx, test_idx = grouped_split_indices(groups, args.seed, args.val_fraction, args.test_fraction)

    train_seq = H5WindowSequence(args.h5, train_idx, args.batch_size, shuffle=True)
    val_seq = H5WindowSequence(args.h5, val_idx, args.batch_size, shuffle=False)
    test_seq = H5WindowSequence(args.h5, test_idx, args.batch_size, shuffle=False)

    model = build_enformer_lite(window_size, bins, tracks, args.channels, args.layers, args.heads, args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(args.out_dir / f"{args.prefix}_history.tsv"), separator="\t"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.out_dir / f"{args.prefix}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(train_seq, validation_data=val_seq, epochs=args.epochs, callbacks=callbacks, verbose=2)
    test = model.evaluate(test_seq, return_dict=True, verbose=1)
    model.save(args.out_dir / f"{args.prefix}_final.keras")

    split_df = pd.concat(
        [
            pd.DataFrame({"window_index": train_idx, "split": "train", "chrom": chrom[train_idx], "subgenome": subgenome[train_idx]}),
            pd.DataFrame({"window_index": val_idx, "split": "val", "chrom": chrom[val_idx], "subgenome": subgenome[val_idx]}),
            pd.DataFrame({"window_index": test_idx, "split": "test", "chrom": chrom[test_idx], "subgenome": subgenome[test_idx]}),
        ],
        ignore_index=True,
    )
    split_df.to_csv(args.out_dir / f"{args.prefix}_splits.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "windows_total", "value": n},
            {"metric": "windows_train", "value": len(train_idx)},
            {"metric": "windows_val", "value": len(val_idx)},
            {"metric": "windows_test", "value": len(test_idx)},
            {"metric": "split_mode", "value": args.split},
            {"metric": "train_chromosomes", "value": int(pd.Series(chrom[train_idx]).nunique())},
            {"metric": "val_chromosomes", "value": int(pd.Series(chrom[val_idx]).nunique())},
            {"metric": "test_chromosomes", "value": int(pd.Series(chrom[test_idx]).nunique())},
            {"metric": "tracks", "value": tracks},
            {"metric": "bins", "value": bins},
            {"metric": "window_size", "value": window_size},
            {"metric": "test_loss", "value": test["loss"]},
            {"metric": "test_mse", "value": test["mse"]},
            {"metric": "best_val_loss", "value": min(history.history["val_loss"])},
        ]
    )
    summary.to_csv(args.out_dir / f"{args.prefix}_summary.tsv", sep="\t", index=False)
    print(json.dumps({str(r.metric): str(r.value) for r in summary.itertuples(index=False)}), flush=True)


if __name__ == "__main__":
    main()
