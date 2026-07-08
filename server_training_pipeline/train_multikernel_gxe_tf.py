from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .trait_isolation import select_single_trait
    from .split_utils import canonical_split_mode, group_kfold_splits, make_split, split_leakage_record
    from .kernel_factorization import effective_factorization_mode, kernel_factors, retained_eigenvalues
except ImportError:
    from trait_isolation import select_single_trait
    from split_utils import canonical_split_mode, group_kfold_splits, make_split, split_leakage_record
    from kernel_factorization import effective_factorization_mode, kernel_factors, retained_eigenvalues


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError:
            fallback = path.with_suffix(".tsv.gz")
            if fallback.exists():
                return pd.read_csv(fallback, sep="\t", low_memory=False)
            raise
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_table(df: pd.DataFrame, path: Path, write_tsv: bool = True) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        write_tsv = True
    if write_tsv:
        df.to_csv(path.with_suffix(".tsv.gz"), sep="\t", index=False)


def persist_and_validate_split_leakage(
    leakage: dict[str, object],
    out_dir: Path,
    prefix: str,
    requested_split: str,
    canonical_split: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([leakage]).to_csv(out_dir / f"{prefix}_split_leakage_qc.tsv", sep="\t", index=False)
    (out_dir / f"{prefix}_split_leakage_qc.json").write_text(
        json.dumps(leakage, default=str, indent=2), encoding="utf-8"
    )
    if leakage["leakage_status"] != "pass":
        raise SystemExit(
            "Split leakage detected. "
            f"requested_split={requested_split!r}; "
            f"canonical_split={canonical_split!r}; "
            f"details={leakage}"
        )


def weighted_mean_std(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0).astype(np.float64)
    y = y.astype(np.float64)
    mu = float(np.sum(w * y) / np.sum(w))
    var = float(np.sum(w * (y - mu) ** 2) / np.sum(w))
    return mu, max(math.sqrt(var), 1e-8)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok].astype(np.float64)
    y_pred = y_pred[ok].astype(np.float64)
    if w is None:
        w = np.ones_like(y_true)
    else:
        w = np.asarray(w, dtype=np.float64)[ok]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.sum(w * err**2) / np.sum(w)))
    mae = float(np.sum(w * np.abs(err)) / np.sum(w))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")
    return {"n": int(len(y_true)), "rmse": rmse, "mae": mae, "pearson": corr}


class LowRankGxE(tf.keras.Model):
    def __init__(
        self,
        g_factors: np.ndarray,
        e_factors: np.ndarray,
        g_rbf_factors: np.ndarray | None = None,
        include_ge: bool = True,
        include_rbf_e: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.G = tf.constant(g_factors, dtype=tf.float32)
        self.E = tf.constant(e_factors, dtype=tf.float32)
        self.G_RBF = tf.constant(g_rbf_factors, dtype=tf.float32) if g_rbf_factors is not None else None
        self.include_ge = include_ge
        self.include_rbf_e = include_rbf_e and g_rbf_factors is not None
        self.weight_decay = float(weight_decay)
        rg = g_factors.shape[1]
        re = e_factors.shape[1]
        rr = g_rbf_factors.shape[1] if g_rbf_factors is not None else 0
        self.intercept = self.add_weight(name="intercept", shape=(), initializer="zeros", trainable=True)
        self.beta_g = self.add_weight(
            name="beta_g",
            shape=(rg,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )
        self.beta_e = self.add_weight(
            name="beta_e",
            shape=(re,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )
        self.beta_g_rbf = (
            self.add_weight(
                name="beta_g_rbf",
                shape=(rr,),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                trainable=True,
            )
            if g_rbf_factors is not None
            else None
        )
        self.beta_ge = (
            self.add_weight(
                name="beta_ge",
                shape=(rg, re),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
                trainable=True,
            )
            if include_ge
            else None
        )
        self.beta_g_rbf_e = (
            self.add_weight(
                name="beta_g_rbf_e",
                shape=(rr, re),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
                trainable=True,
            )
            if self.include_rbf_e
            else None
        )

    def call(self, inputs, training: bool = False):
        gi, ei = inputs
        fg = tf.gather(self.G, gi)
        fe = tf.gather(self.E, ei)
        pred = self.intercept + tf.linalg.matvec(fg, self.beta_g) + tf.linalg.matvec(fe, self.beta_e)
        fr = tf.gather(self.G_RBF, gi) if self.G_RBF is not None else None
        if fr is not None:
            pred = pred + tf.linalg.matvec(fr, self.beta_g_rbf)
        if self.include_ge:
            pred = pred + tf.einsum("br,bs,rs->b", fg, fe, self.beta_ge)
        if self.include_rbf_e:
            pred = pred + tf.einsum("br,bs,rs->b", fr, fe, self.beta_g_rbf_e)
        return pred

    def regularization_loss(self) -> tf.Tensor:
        if self.weight_decay <= 0:
            return tf.constant(0.0, dtype=tf.float32)
        terms = [tf.reduce_sum(self.beta_g**2), tf.reduce_sum(self.beta_e**2)]
        if self.beta_g_rbf is not None:
            terms.append(tf.reduce_sum(self.beta_g_rbf**2))
        if self.beta_ge is not None:
            terms.append(tf.reduce_sum(self.beta_ge**2))
        if self.beta_g_rbf_e is not None:
            terms.append(tf.reduce_sum(self.beta_g_rbf_e**2))
        return self.weight_decay * tf.add_n(terms)


def make_dataset(gi, ei, y, w, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (
            (gi.astype(np.int32), ei.astype(np.int32)),
            y.astype(np.float32),
            w.astype(np.float32),
        )
    )
    if shuffle:
        ds = ds.shuffle(min(len(y), 100_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def evaluate(model: tf.keras.Model, ds: tf.data.Dataset, y_mu: float, y_sd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, pp, ww = [], [], []
    for inputs, y, w in ds:
        pred = model(inputs, training=False).numpy() * y_sd + y_mu
        yy.append(y.numpy() * y_sd + y_mu)
        pp.append(pred)
        ww.append(w.numpy())
    return np.concatenate(yy), np.concatenate(pp), np.concatenate(ww)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--k-g-unique", type=Path, required=True)
    parser.add_argument("--k-g-rbf-unique", type=Path)
    parser.add_argument("--k-e-unique", type=Path, required=True)
    parser.add_argument("--k-g-order", type=Path, required=True)
    parser.add_argument("--k-e-order", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="stage1_mkl_tf")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--rank-g", type=int, default=128)
    parser.add_argument("--rank-g-rbf", type=int, default=128)
    parser.add_argument("--rank-e", type=int, default=64)
    parser.add_argument("--no-ge", action="store_true")
    parser.add_argument("--no-rbf-e", action="store_true")
    parser.add_argument("--split", default="cv2_random_observation")
    parser.add_argument("--group-kfold-col", default="env_kernel_id")
    parser.add_argument("--group-kfold-splits", type=int, default=5)
    parser.add_argument("--group-kfold-fold", type=int, default=0)
    parser.add_argument(
        "--factorization-mode",
        choices=["full_transductive", "train_nystrom"],
        default="full_transductive",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--weight-clip-quantile", type=float, default=0.99)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow: {tf.__version__}; GPUs: {len(gpus)}", flush=True)

    obs = read_table(args.observations)
    obs, selected_trait = select_single_trait(obs, args.trait)
    print(f"Selected trait: {selected_trait}; rows before response filtering: {len(obs):,}", flush=True)
    obs["phenotype_value"] = pd.to_numeric(obs["phenotype_value"], errors="coerce")
    obs["weight_g_e"] = pd.to_numeric(obs["weight_g_e"], errors="coerce")
    obs = obs[obs["phenotype_value"].notna()].copy()
    if obs.empty:
        raise SystemExit(f"Selected trait has zero finite phenotype rows: {selected_trait}")
    print(f"Selected trait: {selected_trait}; finite phenotype rows: {len(obs):,}", flush=True)
    if "original_weight_g_e" not in obs.columns:
        obs["original_weight_g_e"] = obs["weight_g_e"]
    obs["weight_g_e"] = obs["weight_g_e"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    obs["weight_g_e"] = np.where(obs["weight_g_e"] > 0, obs["weight_g_e"], 1.0)
    if args.weight_clip_quantile and 0 < args.weight_clip_quantile < 1:
        cap = float(obs["weight_g_e"].quantile(args.weight_clip_quantile))
        obs["weight_g_e"] = obs["weight_g_e"].clip(upper=cap)

    g_order = pd.read_csv(args.k_g_order, sep="\t")
    e_order = pd.read_csv(args.k_e_order, sep="\t")
    g_map = dict(zip(g_order["source_kernel_index"].astype(int), g_order["compact_kernel_index"].astype(int)))
    e_map = dict(zip(e_order["source_kernel_index"].astype(int), e_order["compact_kernel_index"].astype(int)))
    obs["g_compact"] = obs["geno_kernel_index"].astype(int).map(g_map)
    obs["e_compact"] = obs["env_kernel_index"].astype(int).map(e_map)
    obs = obs[obs["g_compact"].notna() & obs["e_compact"].notna()].reset_index(drop=True)
    obs["g_compact"] = obs["g_compact"].astype(int)
    obs["e_compact"] = obs["e_compact"].astype(int)

    requested_split = args.split
    canonical_split = canonical_split_mode(requested_split, warn=True)
    split_col = {
        "cv2_random_observation": None, "gho_environment": "env_kernel_id", "gho_cycle": "cycle",
        "gho_trial": "trial_name", "gho_country": "country", "gho_family": "canonical_germplasm_key",
        "cv1_genotype": "panel_sample_id", "cv1_environment": "env_kernel_id",
        "cv0_genotype_environment": None, "group_kfold": args.group_kfold_col,
    }[canonical_split]
    if canonical_split == "group_kfold":
        folds = group_kfold_splits(obs, split_col, args.group_kfold_splits, args.seed, args.val_fraction)
        if not 0 <= args.group_kfold_fold < len(folds):
            raise SystemExit(f"--group-kfold-fold must be between 0 and {len(folds) - 1}")
        train_idx, val_idx, test_idx = folds[args.group_kfold_fold]
    else:
        train_idx, val_idx, test_idx = make_split(obs, canonical_split, args.seed, args.test_fraction, args.val_fraction, split_col)
    leakage = split_leakage_record(obs, 0, canonical_split, train_idx, val_idx, test_idx, group_col=split_col)
    persist_and_validate_split_leakage(leakage, args.out_dir, args.prefix, requested_split, canonical_split)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise SystemExit(f"Empty split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    y = obs["phenotype_value"].to_numpy(dtype=np.float32)
    w = obs["weight_g_e"].to_numpy(dtype=np.float32)
    y_mu, y_sd = weighted_mean_std(y[train_idx], w[train_idx])
    y_scaled = ((y - y_mu) / y_sd).astype(np.float32)
    gi = obs["g_compact"].to_numpy(dtype=np.int32)
    ei = obs["e_compact"].to_numpy(dtype=np.int32)

    print("Computing low-rank kernel factors ...", flush=True)
    effective_mode = effective_factorization_mode(args.factorization_mode, canonical_split, warn=True)
    train_g_ids = np.unique(gi[train_idx]) if effective_mode == "train_nystrom" else None
    train_e_ids = np.unique(ei[train_idx]) if effective_mode == "train_nystrom" else None
    Gfac, g_factor_metadata = kernel_factors(args.k_g_unique, args.rank_g, train_g_ids, jitter=1e-6)
    if args.k_g_rbf_unique is not None:
        G_RBF_fac, g_rbf_factor_metadata = kernel_factors(
            args.k_g_rbf_unique, args.rank_g_rbf, train_g_ids, jitter=1e-6
        )
    else:
        G_RBF_fac, g_rbf_factor_metadata = None, None
    Efac, e_factor_metadata = kernel_factors(args.k_e_unique, args.rank_e, train_e_ids, jitter=1e-6)
    gevals = retained_eigenvalues(Gfac, train_g_ids)
    g_rbf_evals = retained_eigenvalues(G_RBF_fac, train_g_ids) if G_RBF_fac is not None else None
    evals = retained_eigenvalues(Efac, train_e_ids)
    model = LowRankGxE(
        Gfac,
        Efac,
        g_rbf_factors=G_RBF_fac,
        include_ge=not args.no_ge,
        include_rbf_e=not args.no_rbf_e,
        weight_decay=args.weight_decay,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_ds = make_dataset(gi[train_idx], ei[train_idx], y_scaled[train_idx], w[train_idx], args.batch_size, True, args.seed)
    val_ds = make_dataset(gi[val_idx], ei[val_idx], y_scaled[val_idx], w[val_idx], args.batch_size, False, args.seed)
    test_ds = make_dataset(gi[test_idx], ei[test_idx], y_scaled[test_idx], w[test_idx], args.batch_size, False, args.seed)

    @tf.function
    def train_step(inputs, yb, wb):
        with tf.GradientTape() as tape:
            pred = model(inputs, training=True)
            loss = tf.reduce_sum(wb * tf.square(pred - yb)) / tf.reduce_sum(wb)
            loss = loss + model.regularization_loss()
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_val = float("inf")
    best_weights = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = []
        for inputs, yb, wb in train_ds:
            losses.append(float(train_step(inputs, yb, wb).numpy()))
        vy, vp, vw = evaluate(model, val_ds, y_mu, y_sd)
        val_m = metrics(vy, vp, vw)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
        if val_m["rmse"] < best_val:
            best_val = val_m["rmse"]
            best_weights = model.get_weights()
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_weights is not None:
        model.set_weights(best_weights)
    ty, tp, tw = evaluate(model, test_ds, y_mu, y_sd)
    vy, vp, vw = evaluate(model, val_ds, y_mu, y_sd)
    test_m = metrics(ty, tp, tw)
    val_m = metrics(vy, vp, vw)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_path = ckpt.save(str(args.out_dir / f"{args.prefix}_ckpt"))
    np.savez_compressed(
        args.out_dir / f"{args.prefix}_kernel_factors_and_scaling.npz",
        Gfac=Gfac,
        G_RBF_fac=G_RBF_fac if G_RBF_fac is not None else np.empty((0, 0), dtype=np.float32),
        Efac=Efac,
        g_eigenvalues=gevals,
        g_rbf_eigenvalues=g_rbf_evals if g_rbf_evals is not None else np.empty(0, dtype=np.float32),
        e_eigenvalues=evals,
        y_mu=np.array([y_mu], dtype=np.float32),
        y_sd=np.array([y_sd], dtype=np.float32),
        requested_factorization_mode=np.array([args.factorization_mode]),
        effective_factorization_mode=np.array([effective_mode]),
        train_genotype_kernel_dimension=np.array([g_factor_metadata["train_kernel_dimension"]], dtype=np.int32),
        train_environment_kernel_dimension=np.array([e_factor_metadata["train_kernel_dimension"]], dtype=np.int32),
        rank_g_requested=np.array([args.rank_g], dtype=np.int32),
        rank_g_retained=np.array([g_factor_metadata["rank_retained"]], dtype=np.int32),
        rank_g_rbf_requested=np.array([args.rank_g_rbf], dtype=np.int32),
        rank_g_rbf_retained=np.array([g_rbf_factor_metadata["rank_retained"] if g_rbf_factor_metadata else 0], dtype=np.int32),
        rank_e_requested=np.array([args.rank_e], dtype=np.int32),
        rank_e_retained=np.array([e_factor_metadata["rank_retained"]], dtype=np.int32),
    )
    pd.DataFrame(history).to_csv(args.out_dir / f"{args.prefix}_training_history.tsv", sep="\t", index=False)

    pred_df = obs.iloc[test_idx].copy()
    pred_df["y_true"] = ty
    pred_df["y_pred"] = tp
    pred_df["split"] = "test"
    write_table(pred_df, args.out_dir / f"{args.prefix}_test_predictions.parquet")

    val_pred_df = obs.iloc[val_idx].copy()
    val_pred_df["y_true"] = vy
    val_pred_df["y_pred"] = vp
    val_pred_df["split"] = "val"
    write_table(val_pred_df, args.out_dir / f"{args.prefix}_val_predictions.parquet")

    summary = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(obs)},
            {"metric": "trait", "value": selected_trait},
            {"metric": "rows_train", "value": len(train_idx)},
            {"metric": "rows_val", "value": len(val_idx)},
            {"metric": "rows_test", "value": len(test_idx)},
            {"metric": "requested_split_mode", "value": requested_split},
            {"metric": "canonical_split_mode", "value": canonical_split},
            {"metric": "split_group_column", "value": split_col or ""},
            {"metric": "split_leakage_status", "value": leakage["leakage_status"]},
            {"metric": "requested_factorization_mode", "value": args.factorization_mode},
            {"metric": "effective_factorization_mode", "value": effective_mode},
            {"metric": "train_genotype_kernel_dimension", "value": g_factor_metadata["train_kernel_dimension"]},
            {"metric": "train_environment_kernel_dimension", "value": e_factor_metadata["train_kernel_dimension"]},
            {"metric": "rank_g_requested", "value": args.rank_g},
            {"metric": "rank_g_retained", "value": g_factor_metadata["rank_retained"]},
            {"metric": "rank_g_rbf_requested", "value": args.rank_g_rbf},
            {"metric": "rank_g_rbf_retained", "value": g_rbf_factor_metadata["rank_retained"] if g_rbf_factor_metadata else 0},
            {"metric": "rank_e_requested", "value": args.rank_e},
            {"metric": "rank_e_retained", "value": e_factor_metadata["rank_retained"]},
            {"metric": "rank_g", "value": Gfac.shape[1]},
            {"metric": "rank_g_rbf", "value": G_RBF_fac.shape[1] if G_RBF_fac is not None else 0},
            {"metric": "rank_e", "value": Efac.shape[1]},
            {"metric": "include_ge", "value": not args.no_ge},
            {"metric": "include_g_rbf", "value": G_RBF_fac is not None},
            {"metric": "include_g_rbf_e", "value": G_RBF_fac is not None and not args.no_rbf_e},
            {"metric": "tensorflow_checkpoint", "value": ckpt_path},
            {"metric": "val_rmse", "value": val_m["rmse"]},
            {"metric": "val_mae", "value": val_m["mae"]},
            {"metric": "val_pearson", "value": val_m["pearson"]},
            {"metric": "test_rmse", "value": test_m["rmse"]},
            {"metric": "test_mae", "value": test_m["mae"]},
            {"metric": "test_pearson", "value": test_m["pearson"]},
        ]
    )
    summary.to_csv(args.out_dir / f"{args.prefix}_summary.tsv", sep="\t", index=False)
    config = vars(args) | {
        "requested_split_mode": requested_split, "canonical_split_mode": canonical_split,
        "split_group_column": split_col or "", "train_rows": len(train_idx), "validation_rows": len(val_idx),
        "test_rows": len(test_idx), "split_leakage_status": leakage["leakage_status"], "selected_trait": selected_trait,
        "requested_factorization_mode": args.factorization_mode, "effective_factorization_mode": effective_mode,
        "train_genotype_kernel_dimension": g_factor_metadata["train_kernel_dimension"],
        "train_environment_kernel_dimension": e_factor_metadata["train_kernel_dimension"],
        "rank_g_retained": g_factor_metadata["rank_retained"],
        "rank_g_rbf_retained": g_rbf_factor_metadata["rank_retained"] if g_rbf_factor_metadata else 0,
        "rank_e_retained": e_factor_metadata["rank_retained"],
    }
    (args.out_dir / f"{args.prefix}_config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
