from modeling.EGNN import EGNN
import equinox as eqx
import jax.numpy as jnp
from typing import Optional

class Predictor(eqx.Module):
    gnn: EGNN

    def __init__(self, key, cfg):
        # The Predictor is essentially another EGNN
        # It maps z_t (latent) back to z_t+dt (latent)
        # We reuse the same config but ensure in_feat_dim matches z_dim
        self.gnn = EGNN(key, cfg)

    def __call__(self, x_t: jnp.ndarray, z_t: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        """
        x_t: Coordinates at time t (N, d)
        z_t: Latent embeddings from the Encoder (N, h_dim)
        """
        # We predict the CHANGE in z (residual) or the absolute next z
        # In JEPA, predicting the next absolute state is common
        delta_z, _ = self.gnn(x_t, z_t, mask=mask)
        
        return delta_z