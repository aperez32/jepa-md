import os
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import umap

from modeling.EGNN import EGNN
from modeling.predictor import Predictor
from train.train import JEPA
from dataloader.dataloader import lj_pair_batches


def is_trainable(x):
    return eqx.is_inexact_array(x)


def extract_student_latents(
    model,
    npz_path,
    batch_size,
    mask_ratio,
    num_batches=200,
    seed=0,
    normalize_time=True,
):
    """
    Returns:
      Z: (M, D) numpy array of node latents
      C: (M,) numpy array of timestep indices
    """
    it = lj_pair_batches(
        npz_path,
        batch_size=batch_size,
        mask_ratio=mask_ratio,
        shuffle=False,   # IMPORTANT: keep temporal order
        seed=seed,
    )

    Z_list = []
    C_list = []

    for t in range(num_batches):
        try:
            x_t, h_t_m, mask, x_tp1, h_tp1 = next(it)
        except StopIteration:
            break

        enc_s = jax.vmap(model.student, in_axes=(0, 0, 0), out_axes=(0, 0))
        z_t, _ = enc_s(x_t, h_t_m, mask)  # (B, N, D)

        z_np = np.array(z_t).reshape(-1, z_t.shape[-1])  # (B*N, D)
        Z_list.append(z_np)

        # timestep color (one value per node)
        t_color = np.full(z_np.shape[0], t, dtype=np.float32)
        C_list.append(t_color)

    Z = np.concatenate(Z_list, axis=0)
    C = np.concatenate(C_list, axis=0)

    if normalize_time and C.size > 0:
        C = (C - C.min()) / (C.max() - C.min() + 1e-8)

    return Z, C

def plot_scatter(Y, title, out_path, c=None, s=3, alpha=0.4):
    plt.figure(figsize=(6, 6))
    if c is None:
        plt.scatter(Y[:, 0], Y[:, 1], s=s, alpha=alpha)
    else:
        plt.scatter(Y[:, 0], Y[:, 1], c=c, s=s, alpha=alpha, cmap="magma")
        plt.colorbar(label="timestep")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(cfg, contextcfg, predcfg, seed, out_dir, ckpt_dir):
    
    # ---- Rebuild skeleton for loading ----
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)

    student = EGNN(k1, cfg=contextcfg)
    pred = Predictor(k2, cfg=predcfg)
    model = JEPA(student=student, pred=pred)
    teacher = jax.tree_util.tree_map(lambda x: x, student)

    like = {
        "model": model,
        "teacher": teacher,
        "epoch": 0,
        "val_loss": 0.0,
    }

    ckpt = eqx.tree_deserialise_leaves(f"{ckpt_dir}", like)
    model = ckpt["model"]
    teacher = ckpt["teacher"]
    epoch = ckpt["epoch"]
    val_loss = ckpt["val_loss"]
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss}")

    # ---- Extract latents from validation trajectory ----
    val_path = cfg.val_fname
    Z, C = extract_student_latents(
        model,
        npz_path=f'{out_dir}/{val_path}',
        batch_size=cfg.batch_size,
        mask_ratio=cfg.mask_ratio,
        num_batches=getattr(cfg, "probe_batches", 200),
        seed=0,
    )

    print("Latents matrix:", Z.shape)

    # Optionally subsample for speed (UMAP can be slow)
    max_points = getattr(cfg, "probe_max_points", 20000)
    if Z.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(Z.shape[0], size=max_points, replace=False)
        Z = Z[idx]
        if C is not None:
            C = C[idx]
        print("Subsampled latents to:", Z.shape)

    # ---- PCA ----
    pca = PCA(n_components=2, random_state=0)
    Y_pca = pca.fit_transform(Z)
    plot_scatter(Y_pca, f"PCA", f"{out_dir}/pca_latents.png", c=C)

    # ---- UMAP ----
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",     # cosine often works well for JEPA-like latents
        random_state=0,
    )
    Y_umap = reducer.fit_transform(Z)
    plot_scatter(Y_umap, f"UMAP", f"{out_dir}/umap_latents.png", c=C)

    print("Wrote: evals/pca_latents.png and eval/umap_latents.png")

