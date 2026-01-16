import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from modeling.EGNN import EGNN
from modeling.predictor import Predictor
from dataloader.dataloader import lj_pair_batches
from train.train_utils import ema_update, l2_normalize, masked_cosine_loss, masked_cosine_sim, eval_step, save_checkpoint
import matplotlib.pyplot as plt

class JEPA(eqx.Module):
    student: any
    pred: any


def main(cfg, contextcfg, predcfg, seed, out_dir):


    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)

    student = EGNN(k1, cfg=contextcfg)
    pred = Predictor(k2, cfg=predcfg)
    model = JEPA(student=student, pred=pred)

    # teacher = copy of student at init
    teacher = jax.tree_util.tree_map(lambda x: x, student)

    def is_trainable(x):
        return eqx.is_inexact_array(x)

    opt = optax.adam(cfg.lr)
    opt_state = opt.init(eqx.filter(model, is_trainable))
    ema_decay = getattr(cfg, "ema_decay", 0.996)

    def ema_update(teacher, student, decay):
        def _ema(t, s):
            if eqx.is_inexact_array(t):
                return decay * t + (1.0 - decay) * s
            return t
        return jax.tree_util.tree_map(_ema, teacher, student)

    @eqx.filter_jit
    def train_step(model, teacher, opt_state, batch):
        x_t, h_t_m, mask, x_tp1, h_tp1 = batch
        mask_none = jnp.zeros_like(mask)

        def loss_and_metrics(model):
            enc_s = jax.vmap(model.student, in_axes=(0,0,0), out_axes=(0,0))
            enc_t = jax.vmap(teacher,       in_axes=(0,0,0), out_axes=(0,0))

            z_t, _   = enc_s(x_t,  h_t_m, mask)                 # (B,N,D)
            z_tp1, _ = enc_t(x_tp1, h_tp1, mask_none)           # (B,N,D)
            z_tp1 = jax.lax.stop_gradient(z_tp1)

            pred_delta_z = jax.vmap(model.pred)(x_t, z_t)        # (B,N,D)
            z_pred = z_t + pred_delta_z

            loss = masked_cosine_loss(z_pred, z_tp1, mask)

            var_z_t = jnp.mean(jnp.var(z_t, axis=(0, 1)))         # average over D

            # mean norms
            mean_norm_z_t   = jnp.mean(jnp.linalg.norm(z_t, axis=-1))    # over B,N
            mean_norm_z_tp1 = jnp.mean(jnp.linalg.norm(z_tp1, axis=-1))  # over B,N

            # cosine similarity on masked nodes (higher is better)
            cos_masked = masked_cosine_sim(z_pred, z_tp1, mask)

            metrics = {
                "var_z_t": var_z_t,
                "mean_norm_z_t": mean_norm_z_t,
                "mean_norm_z_tp1": mean_norm_z_tp1,
                "cos_masked": cos_masked,
            }
            return loss, metrics

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_and_metrics, has_aux=True)(model)

        grads_f  = eqx.filter(grads, is_trainable)
        params_f = eqx.filter(model, is_trainable)

        updates, opt_state = opt.update(grads_f, opt_state, params=params_f)
        model = eqx.apply_updates(model, updates)

        teacher = ema_update(teacher, model.student, ema_decay)
        return model, teacher, opt_state, loss, metrics

    t_losses = []
    v_losses = []
    for epoch in range(1, cfg.num_epochs+1):
        # ---- train ----
        train_it = lj_pair_batches(
            f'{out_dir}/{cfg.train_fname}',
            batch_size=cfg.batch_size,
            mask_ratio=cfg.mask_ratio,
            shuffle=False,
            seed=epoch,
        )

        train_losses = []
        for batch in train_it:
            model, teacher, opt_state, loss, metrics = train_step(model, teacher, opt_state, batch)
            train_losses.append(loss)

        train_loss = float(jnp.mean(jnp.stack(train_losses))) if train_losses else float("nan")

        # ---- val ----
        val_it = lj_pair_batches(
            f'{out_dir}/{cfg.val_fname}',
            batch_size=cfg.batch_size,
            mask_ratio=cfg.mask_ratio,
            shuffle=False,
            seed=0,
        )

        vals = []
        for batch in val_it:
            vals.append(eval_step(model, teacher, batch))

        if vals:
            vals = jnp.stack(vals)  # (num_batches, 5)
            val_loss = float(jnp.mean(vals[:, 0]))
            val_varz = float(jnp.mean(vals[:, 1]))
            val_nzt  = float(jnp.mean(vals[:, 2]))
            val_nz1  = float(jnp.mean(vals[:, 3]))
            val_cos  = float(jnp.mean(vals[:, 4]))
        else:
            val_loss = val_varz = val_nzt = val_nz1 = val_cos = float("nan")

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f} | "
            f"val var(z_t) {val_varz:.3e} | "
            f"val ||z_t|| {val_nzt:.4f} | "
            f"val ||z_tp1|| {val_nz1:.4f} | "
            f"val cos_masked {val_cos:.4f}"
        )
        t_losses.append(train_loss)
        v_losses.append(val_loss)

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                path=f"{out_dir}/checkpoints/epoch_{epoch}.eqx",
                model=model,
                teacher=teacher,
                epoch=epoch,
                val_loss=val_loss,
            )
    plt.plot(range(cfg.num_epochs), t_losses, label='Training loss')
    plt.plot(range(cfg.num_epochs), v_losses, label='Validation loss')
    plt.legend()
    plt.savefig(f"{out_dir}/loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    
        


if __name__ == "__main__":
    main()
