import jax
import equinox as eqx

# mlp helper
def _mlp(key, in_dim, hidden_dim, out_dim, depth=2, act=jax.nn.silu):
    keys = jax.random.split(key, depth + 1)
    layers = []
    d = in_dim
    for i in range(depth):
        layers.append(eqx.nn.Linear(d, hidden_dim, key=keys[i]))
        d = hidden_dim
    layers.append(eqx.nn.Linear(d, out_dim, key=keys[-1]))
    return layers, act


def _apply_mlp(layers, act, x):
    for layer in layers[:-1]:
        x = act(layer(x))
    return layers[-1](x)
