import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from modeling.utils import _mlp, _apply_mlp


class EGNNLayer(eqx.Module):
    # MLPs for edge messages and node update and coordinate scaling
    edge_mlp_layers: list
    node_mlp_layers: list
    coord_mlp_layers: list
    edge_act: callable
    node_act: callable
    coord_act: callable

    residual_h: bool = True

    def __init__(
        self,
        key,
        h_dim: int,
        msg_dim: int,
        hidden_dim: int = 128,
        mlp_depth: int = 2,
        residual_h: bool = True,
    ):
        k1, k2, k3 = jax.random.split(key, 3)

        # Edge MLP input: [h_i, h_j, r2] where r2 is scalar
        self.edge_mlp_layers, self.edge_act = _mlp(
            k1, in_dim=2 * h_dim + 1, hidden_dim=hidden_dim, out_dim=msg_dim, depth=mlp_depth
        )

        # Node MLP input: [h_i, sum_j m_ij]
        self.node_mlp_layers, self.node_act = _mlp(
            k2, in_dim=h_dim + msg_dim, hidden_dim=hidden_dim, out_dim=h_dim, depth=mlp_depth
        )

        # Coord MLP maps message -> scalar coefficient for coord update
        # (one scalar per edge, to scale (x_i - x_j))
        self.coord_mlp_layers, self.coord_act = _mlp(
            k3, in_dim=msg_dim, hidden_dim=hidden_dim, out_dim=1, depth=mlp_depth
        )

        self.residual_h = residual_h

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        """
        x: (N, d) positions
        h: (N, h_dim) node features (scalars)
        mask: optional (N,) bool for active nodes (for padding). For your fixed N=16, you can omit.
        Returns: x_new, h_new
        """
        N, d = x.shape

        # Pairwise differences x_i - x_j: shape (N, N, d)
        x_i = x[:, None, :]          # (N,1,d)
        x_j = x[None, :, :]          # (1,N,d)
        dx = x_i - x_j               # (N,N,d)
        r2 = jnp.sum(dx * dx, axis=-1, keepdims=True)  # (N,N,1)

        # Build edge inputs: [h_i, h_j, r2]
        # Build edge inputs: [h_i, h_j, r2]
        h_i = h[:, None, :]          # (N,1,h)
        h_j = h[None, :, :]          # (1,N,h)

        # Explicitly broadcast h_i and h_j to (N, N, h)
        # jnp.broadcast_to is efficient and doesn't actually copy data in memory
        h_i_bp = jnp.broadcast_to(h_i, (N, N, h.shape[-1]))
        h_j_bp = jnp.broadcast_to(h_j, (N, N, h.shape[-1]))

        # Now all shapes match on axes 0 and 1: (N, N, h), (N, N, h), (N, N, 1)
        edge_in = jnp.concatenate([h_i_bp, h_j_bp, r2], axis=-1)  # Result: (N,N,2h+1)

        # Compute messages m_ij
        # vmap over (N,N,*) implicitly by applying MLP to last axis
        m_ij = jax.vmap(jax.vmap(lambda e: _apply_mlp(self.edge_mlp_layers, self.edge_act, e)))(edge_in)
        # m_ij: (N,N,msg_dim)

        # Remove self edges
        not_self = 1.0 - jnp.eye(N, dtype=x.dtype)  # (N,N)
        m_ij = m_ij * not_self[:, :, None]

        # Optional node mask (for padded batches)
        if mask is not None:
            mask_f = mask.astype(x.dtype)
            m_ij = m_ij * (mask_f[:, None, None] * mask_f[None, :, None])

        # Coordinate update: x_i <- x_i + sum_j (x_i - x_j) * phi(m_ij)
        # phi(m_ij) is scalar (N,N,1)
        phi_ij = jax.vmap(jax.vmap(lambda m: _apply_mlp(self.coord_mlp_layers, self.coord_act, m)))(m_ij)
        # Stabilize (optional): scale by 1/N
        x_update = jnp.sum(dx * phi_ij, axis=1) / (N - 1 + 1e-8)   # (N,d)
        x_new = x + x_update

        # Node update: h_i <- h_i + MLP([h_i, sum_j m_ij])
        m_i = jnp.sum(m_ij, axis=1)  # (N,msg_dim)
        node_in = jnp.concatenate([h, m_i], axis=-1)  # (N,h+msg)
        dh = jax.vmap(lambda u: _apply_mlp(self.node_mlp_layers, self.node_act, u))(node_in)
        h_new = h + dh if self.residual_h else dh

        if mask is not None:
            # Keep masked nodes unchanged
            x_new = jnp.where(mask[:, None], x_new, x)
            h_new = jnp.where(mask[:, None], h_new, h)

        return x_new, h_new


class EGNN(eqx.Module):
    layers: list
    h_in: eqx.nn.Linear
    h_out: eqx.nn.Linear

    def __init__(self, key, cfg):
        
        k_in, k_layers, k_out = jax.random.split(key, 3)
        self.h_in = eqx.nn.Linear(cfg.in_feat_dim, cfg.h_dim, key=k_in)
        self.h_out = eqx.nn.Linear(cfg.h_dim, cfg.out_dim, key=k_out)

        keys = jax.random.split(k_layers, cfg.n_layers)
        self.layers = [
            EGNNLayer(
                key=keys[i],
                h_dim=cfg.h_dim,
                msg_dim=cfg.msg_dim,
                hidden_dim=cfg.hidden_dim,
                mlp_depth=cfg.mlp_depth,
                residual_h=True,
            )
            for i in range(cfg.n_layers)
        ]

    def __call__(self, x: jnp.ndarray, node_feats: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        """
        x: (N,d)
        node_feats: (N,in_feat_dim)  (e.g., could be velocities, speeds, or other per-particle features)
        returns:
          z_nodes: (N,out_dim) node embeddings
          x_latent: (N,d) updated coords (often you ignore this for encoding, but it's there)
        """
        h = jax.vmap(self.h_in)(node_feats)
        for layer in self.layers:
            x, h = layer(x, h, mask=mask)
        z = jax.vmap(self.h_out)(h)
        return z, x


