from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from trait_isolation import select_single_trait


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


def weighted_mean_std(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0).astype(np.float64)
    y = y.astype(np.float64)
    mu = float(np.sum(w * y) / np.sum(w))
    var = float(np.sum(w * (y - mu) ** 2) / np.sum(w))
    return mu, max(math.sqrt(var), 1e-8)


def top_kernel_factors(kernel_path: Path | np.ndarray, rank: int, jitter: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    K = np.load(kernel_path).astype(np.float64) if isinstance(kernel_path, Path) else np.asarray(kernel_path, dtype=np.float64)
    K = (K + K.T) / 2.0
    if jitter > 0:
        K.flat[:: K.shape[0] + 1] += jitter
    vals, vecs = np.linalg.eigh(K)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > 1e-8
    vals = vals[keep][:rank]
    vecs = vecs[:, keep][:, :rank]
    factors = vecs * np.sqrt(vals)[None, :]
    return factors.astype(np.float32), vals.astype(np.float32)


def aligned_optional_kernel(kernel_path: Path, order_path: Path, compact_g_order_path: Path) -> np.ndarray:
    K = np.load(kernel_path, mmap_mode="r")
    order = pd.read_csv(order_path, sep="\t", dtype=str)
    compact = pd.read_csv(compact_g_order_path, sep="\t", dtype=str)
    id_candidates = ["sample_id", "panel_sample_id", "canonical_germplasm_key"]
    order_col = next((c for c in id_candidates if c in order.columns), None)
    compact_col = next((c for c in id_candidates if c in compact.columns), None)
    if order_col is None or compact_col is None:
        raise SystemExit(f"Could not identify genotype ID columns for optional kernel: {order_path}, {compact_g_order_path}")
    ids = order[order_col].astype(str).str.strip()
    if ids.duplicated().any() or K.ndim != 2 or K.shape[0] != K.shape[1] or len(ids) != K.shape[0]:
        raise SystemExit(f"Optional kernel/order mismatch or duplicated IDs: {kernel_path}, {order_path}, shape={K.shape}, order={len(ids)}")
    mapper = {sample_id: i for i, sample_id in enumerate(ids)}
    wanted = compact[compact_col].astype(str).str.strip()
    missing = wanted[~wanted.isin(mapper)]
    if not missing.empty:
        raise SystemExit(f"Optional kernel {kernel_path} misses {len(missing)} compact model genotypes; examples={missing.head(5).tolist()}")
    idx = wanted.map(mapper).to_numpy(dtype=np.int64)
    return np.asarray(K[np.ix_(idx, idx)], dtype=np.float32)


def split_indices(
    df: pd.DataFrame,
    mode: str,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    group_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(df)
    if mode == "random":
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]

    if group_col not in df.columns:
        raise SystemExit(f"Requested split mode {mode}, but group column is absent: {group_col}")
    groups = df[group_col].astype(str).fillna("").unique()
    rng.shuffle(groups)
    n_test_groups = max(1, int(round(len(groups) * test_fraction)))
    n_val_groups = max(1, int(round(len(groups) * val_fraction)))
    test_groups = set(groups[:n_test_groups])
    val_groups = set(groups[n_test_groups : n_test_groups + n_val_groups])
    group_series = df[group_col].astype(str).fillna("")
    test = np.where(group_series.isin(test_groups))[0]
    val = np.where(group_series.isin(val_groups))[0]
    train = np.where(~group_series.isin(test_groups | val_groups))[0]
    return train, val, test


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
        a_factors: np.ndarray | None = None,
        z_factors: np.ndarray | None = None,
        include_ge: bool = True,
        include_rbf_e: bool = True,
        include_ae: bool = True,
        include_ze: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.G = tf.constant(g_factors, dtype=tf.float32)
        self.E = tf.constant(e_factors, dtype=tf.float32)
        self.G_RBF = tf.constant(g_rbf_factors, dtype=tf.float32) if g_rbf_factors is not None else None
        self.A = tf.constant(a_factors, dtype=tf.float32) if a_factors is not None else None
        self.Z = tf.constant(z_factors, dtype=tf.float32) if z_factors is not None else None
        self.include_ge = include_ge
        self.include_rbf_e = include_rbf_e and g_rbf_factors is not None
        self.include_ae = include_ae and a_factors is not None
        self.include_ze = include_ze and z_factors is not None
        self.weight_decay = float(weight_decay)
        rg = g_factors.shape[1]
        re = e_factors.shape[1]
        rr = g_rbf_factors.shape[1] if g_rbf_factors is not None else 0
        ra = a_factors.shape[1] if a_factors is not None else 0
        rz = z_factors.shape[1] if z_factors is not None else 0
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
        self.beta_a = self.add_weight(name="beta_a", shape=(ra,), initializer=tf.keras.initializers.RandomNormal(stddev=0.01), trainable=True) if a_factors is not None else None
        self.beta_z = self.add_weight(name="beta_z", shape=(rz,), initializer=tf.keras.initializers.RandomNormal(stddev=0.01), trainable=True) if z_factors is not None else None
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
        self.beta_ae = self.add_weight(name="beta_ae", shape=(ra, re), initializer=tf.keras.initializers.RandomNormal(stddev=0.001), trainable=True) if self.include_ae else None
        self.beta_ze = self.add_weight(name="beta_ze", shape=(rz, re), initializer=tf.keras.initializers.RandomNormal(stddev=0.001), trainable=True) if self.include_ze else None

    def call(self, inputs, training: bool = False):
        gi, ei = inputs
        fg = tf.gather(self.G, gi)
        fe = tf.gather(self.E, ei)
        pred = self.intercept + tf.linalg.matvec(fg, self.beta_g) + tf.linalg.matvec(fe, self.beta_e)
        fr = tf.gather(self.G_RBF, gi) if self.G_RBF is not None else None
        if fr is not None:
            pred = pred + tf.linalg.matvec(fr, self.beta_g_rbf)
        fa = tf.gather(self.A, gi) if self.A is not None else None
        fz = tf.gather(self.Z, gi) if self.Z is not None else None
        if fa is not None:
            pred = pred + tf.linalg.matvec(fa, self.beta_a)
        if fz is not None:
            pred = pred + tf.linalg.matvec(fz, self.beta_z)
        if self.include_ge:
            pred = pred + tf.einsum("br,bs,rs->b", fg, fe, self.beta_ge)
        if self.include_rbf_e:
            pred = pred + tf.einsum("br,bs,rs->b", fr, fe, self.beta_g_rbf_e)
        if self.include_ae:
            pred = pred + tf.einsum("br,bs,rs->b", fa, fe, self.beta_ae)
        if self.include_ze:
            pred = pred + tf.einsum("br,bs,rs->b", fz, fe, self.beta_ze)
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
        for optional in (self.beta_a, self.beta_z, self.beta_ae, self.beta_ze):
            if optional is not None:
                terms.append(tf.reduce_sum(optional**2))
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
    parser.add_argument("--k-a", type=Path)
    parser.add_argument("--k-a-order", type=Path)
    parser.add_argument("--k-z", type=Path)
    parser.add_argument("--k-z-order", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="stage1_mkl_tf")
    parser.add_argument("--trait", action="append")
    parser.add_argument("--rank-g", type=int, default=128)
    parser.add_argument("--rank-g-rbf", type=int, default=128)
    parser.add_argument("--rank-e", type=int, default=64)
    parser.add_argument("--rank-a", type=int, default=64)
    parser.add_argument("--rank-z", type=int, default=64)
    parser.add_argument("--no-ge", action="store_true")
    parser.add_argument("--no-rbf-e", action="store_true")
    parser.add_argument("--no-ae", action="store_true")
    parser.add_argument("--no-ze", action="store_true")
    parser.add_argument("--split", choices=["random", "loeo", "loyo", "loco"], default="loeo")
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

    split_col = {"random": "", "loeo": "env_kernel_id", "loyo": "cycle", "loco": "country"}[args.split]
    train_idx, val_idx, test_idx = split_indices(obs, args.split, args.test_fraction, args.val_fraction, args.seed, split_col)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise SystemExit(f"Empty split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    y = obs["phenotype_value"].to_numpy(dtype=np.float32)
    w = obs["weight_g_e"].to_numpy(dtype=np.float32)
    y_mu, y_sd = weighted_mean_std(y[train_idx], w[train_idx])
    y_scaled = ((y - y_mu) / y_sd).astype(np.float32)
    gi = obs["g_compact"].to_numpy(dtype=np.int32)
    ei = obs["e_compact"].to_numpy(dtype=np.int32)

    print("Computing low-rank kernel factors ...", flush=True)
    Gfac, gevals = top_kernel_factors(args.k_g_unique, args.rank_g)
    G_RBF_fac, g_rbf_evals = (
        top_kernel_factors(args.k_g_rbf_unique, args.rank_g_rbf)
        if args.k_g_rbf_unique is not None
        else (None, None)
    )
    Efac, evals = top_kernel_factors(args.k_e_unique, args.rank_e)
    Afac = aevals = None
    Zfac = zevals = None
    if args.k_a or args.k_a_order:
        if not args.k_a or not args.k_a_order:
            raise SystemExit("Provide both --k-a and --k-a-order")
        K_A = aligned_optional_kernel(args.k_a, args.k_a_order, args.k_g_order)
        Afac, aevals = top_kernel_factors(K_A, args.rank_a)
    if args.k_z or args.k_z_order:
        if not args.k_z or not args.k_z_order:
            raise SystemExit("Provide both --k-z and --k-z-order")
        K_Z = aligned_optional_kernel(args.k_z, args.k_z_order, args.k_g_order)
        Zfac, zevals = top_kernel_factors(K_Z, args.rank_z)
    model = LowRankGxE(
        Gfac,
        Efac,
        g_rbf_factors=G_RBF_fac,
        a_factors=Afac,
        z_factors=Zfac,
        include_ge=not args.no_ge,
        include_rbf_e=not args.no_rbf_e,
        include_ae=not args.no_ae,
        include_ze=not args.no_ze,
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
        Afac=Afac if Afac is not None else np.empty((0, 0), dtype=np.float32),
        Zfac=Zfac if Zfac is not None else np.empty((0, 0), dtype=np.float32),
        g_eigenvalues=gevals,
        g_rbf_eigenvalues=g_rbf_evals if g_rbf_evals is not None else np.empty(0, dtype=np.float32),
        e_eigenvalues=evals,
        a_eigenvalues=aevals if aevals is not None else np.empty(0, dtype=np.float32),
        z_eigenvalues=zevals if zevals is not None else np.empty(0, dtype=np.float32),
        y_mu=np.array([y_mu], dtype=np.float32),
        y_sd=np.array([y_sd], dtype=np.float32),
    )
    pd.DataFrame(history).to_csv(args.out_dir / f"{args.prefix}_training_history.tsv", sep="\t", index=False)

    pred_df = obs.iloc[test_idx].copy()
    pred_df["y_true"] = ty
    pred_df["y_pred"] = tp
    pred_df["split"] = "test"
    write_table(pred_df, args.out_dir / f"{args.prefix}_test_predictions.parquet")

    summary = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(obs)},
            {"metric": "trait", "value": selected_trait},
            {"metric": "rows_train", "value": len(train_idx)},
            {"metric": "rows_val", "value": len(val_idx)},
            {"metric": "rows_test", "value": len(test_idx)},
            {"metric": "split", "value": args.split},
            {"metric": "rank_g", "value": Gfac.shape[1]},
            {"metric": "rank_g_rbf", "value": G_RBF_fac.shape[1] if G_RBF_fac is not None else 0},
            {"metric": "rank_e", "value": Efac.shape[1]},
            {"metric": "rank_a", "value": Afac.shape[1] if Afac is not None else 0},
            {"metric": "rank_z", "value": Zfac.shape[1] if Zfac is not None else 0},
            {"metric": "include_ge", "value": not args.no_ge},
            {"metric": "include_g_rbf", "value": G_RBF_fac is not None},
            {"metric": "include_g_rbf_e", "value": G_RBF_fac is not None and not args.no_rbf_e},
            {"metric": "include_a", "value": Afac is not None},
            {"metric": "include_ae", "value": Afac is not None and not args.no_ae},
            {"metric": "include_z", "value": Zfac is not None},
            {"metric": "include_ze", "value": Zfac is not None and not args.no_ze},
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
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
