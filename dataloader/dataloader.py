import numpy as np
import jax
import jax.numpy as jnp

def build_features(pos, vel): # Build features
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)  # (T,N,1)
    h = jnp.concatenate([vel, speed], axis=-1)            # (T,N,d+1)
    return h

def spatial_mask(key, x, mask_ratio):
    
    N = x.shape[0]
    k1, k2 = jax.random.split(key, 2)
    center_idx = jax.random.randint(k1, (), 0, N)
    center = x[center_idx]
    dist = jnp.linalg.norm(x - center[None, :], axis=-1)   # (N,)
    r = jnp.quantile(dist, mask_ratio)                     # radius so ~mask_ratio masked
    return dist <= r

def apply_mask_to_h(h, mask):

    mask_f = mask[:, None]
    h0 = jnp.where(mask_f, 0.0, h)
    return jnp.concatenate([h0, mask_f.astype(h.dtype)], axis=-1)

def lj_pair_batches(npz_path, batch_size, mask_ratio=0.25, shuffle=True, seed=0):

    data = np.load(npz_path)
    pos = jnp.asarray(data["pos"])  # (T,N,d)
    vel = jnp.asarray(data["vel"])  # (T,N,d)

    T, N, d = pos.shape
    h = build_features(pos, vel)    # (T,N,f), f=d+1

    num_pairs = T - 1
    key = jax.random.PRNGKey(seed)

    # indices t for pairs (t, t+1)
    idx = jnp.arange(num_pairs)
    if shuffle:
        key, k = jax.random.split(key)
        idx = jax.random.permutation(k, idx)

    # iterate batches
    for start in range(0, num_pairs, batch_size):
        bidx = idx[start:start + batch_size]       # (B,)
        x_t = pos[bidx]                            # (B,N,d)
        h_t = h[bidx]                              # (B,N,f)
        x_tp1 = pos[bidx + 1]                      # (B,N,d)
        h_tp1 = h[bidx + 1]                        # (B,N,f)

        # make one spatial mask per sample
        key, km = jax.random.split(key)
        keys = jax.random.split(km, bidx.shape[0])

        mask = jax.vmap(spatial_mask, in_axes=(0, 0, None))(keys, x_t, mask_ratio)  # (B,N)

        # context features: zero masked nodes + append mask flag -> (B,N,f+1)
        h_t_m = jax.vmap(apply_mask_to_h)(h_t, mask)

        # target features: append ZERO mask flag -> (B,N,f+1)
        zero_flag = jnp.zeros((*h_tp1.shape[:2], 1), dtype=h_tp1.dtype)
        h_tp1_m0 = jnp.concatenate([h_tp1, zero_flag], axis=-1)

        yield x_t, h_t_m, mask, x_tp1, h_tp1_m0
