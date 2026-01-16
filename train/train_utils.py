import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import os


def ema_update(teacher, student, decay):
    def _ema(t, s):
        if eqx.is_array(t):
            return decay * t + (1.0 - decay) * s
        return t
    return jax.tree_util.tree_map(_ema, teacher, student)

def l2_normalize(x, eps=1e-8):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

def masked_cosine_loss(z_pred, z_targ, mask):
    z_pred = l2_normalize(z_pred)
    z_targ = l2_normalize(z_targ)
    cos = jnp.sum(z_pred * z_targ, axis=-1)   # (B,N)
    loss_node = 1.0 - cos                     # (B,N)
    w = mask.astype(loss_node.dtype)
    return jnp.sum(loss_node * w) / (jnp.sum(w) + 1e-8)

def masked_cosine_sim(z_a, z_b, mask):
    """Mean cosine similarity on masked nodes only."""
    z_a = l2_normalize(z_a)
    z_b = l2_normalize(z_b)
    cos = jnp.sum(z_a * z_b, axis=-1)         # (B,N)
    w = mask.astype(cos.dtype)
    return jnp.sum(cos * w) / (jnp.sum(w) + 1e-8)

@eqx.filter_jit
def eval_step(model, teacher, batch):
    x_t, h_t_m, mask, x_tp1, h_tp1 = batch
    mask_none = jnp.zeros_like(mask)

    enc_s = jax.vmap(model.student, in_axes=(0,0,0), out_axes=(0,0))
    enc_t = jax.vmap(teacher,       in_axes=(0,0,0), out_axes=(0,0))

    z_t, _   = enc_s(x_t,  h_t_m, mask)
    z_tp1, _ = enc_t(x_tp1, h_tp1, mask_none)
    z_tp1 = jax.lax.stop_gradient(z_tp1)

    pred_delta_z = jax.vmap(model.pred)(x_t, z_t)
    z_pred = z_t + pred_delta_z

    loss = masked_cosine_loss(z_pred, z_tp1, mask)

    # metrics
    var_z_t = jnp.mean(jnp.var(z_t, axis=(0, 1)))
    mean_norm_z_t = jnp.mean(jnp.linalg.norm(z_t, axis=-1))
    mean_norm_z_tp1 = jnp.mean(jnp.linalg.norm(z_tp1, axis=-1))
    cos_masked = masked_cosine_sim(z_pred, z_tp1, mask)

    return jnp.stack([loss, var_z_t, mean_norm_z_t, mean_norm_z_tp1, cos_masked])

def save_checkpoint(path, model, teacher, epoch, val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    eqx.tree_serialise_leaves(
        path,
        {
            "model": model,
            "teacher": teacher,
            "epoch": epoch,
            "val_loss": val_loss,
        },
    )
