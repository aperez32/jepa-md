import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space, energy, simulate, quantity
import matplotlib.pyplot as plt


def sample_positions_min_sep(key, N, dim, box_size, min_sep, max_tries=20000):
    """
    Simple sequential rejection sampler with periodic boundary conditions.
    Returns R0 with all pair distances >= min_sep (up to max_tries).
    """
    # Use numpy RNG seeded from JAX key (easiest outside jit)
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    R = np.empty((N, dim), dtype=jnp.float64)
    placed = 0
    tries = 0

    def pbc_delta(a, b):
        d = a - b
        d -= box_size * np.round(d / box_size)
        return d

    while placed < N and tries < max_tries:
        tries += 1
        cand = rng.uniform(0.0, box_size, size=(dim,))

        ok = True
        for j in range(placed):
            d = pbc_delta(cand, R[j])
            if np.linalg.norm(d) < min_sep:
                ok = False
                break

        if ok:
            R[placed] = cand
            placed += 1

    if placed < N:
        raise RuntimeError(
            f"Failed to place all particles with min_sep={min_sep}. "
            f"Try lowering min_sep or N, or increasing box_size/max_tries."
        )

    return jnp.array(R, dtype=jnp.float32)

def main(cfg, seed, out_dir, is_train=True):
    key = jax.random.PRNGKey(seed)

    N = cfg.N
    dim = cfg.dim
    box_size = cfg.box_size
    dt = cfg.dt
    num_steps = cfg.num_steps
    fname = cfg.train_fname if is_train else cfg.val_fname

    mass = cfg.mass
    kT = cfg.kT
    sigma = cfg.sigma
    epsilon = cfg.epsilon

    displacement, shift = space.periodic(box_size)

    lj_energy = energy.lennard_jones_pair(
        displacement,
        sigma=sigma,
        epsilon=epsilon,
        r_onset=2.0 * sigma,
        r_cutoff=2.5 * sigma
    )

    key, k1, k2 = jax.random.split(key, 3)
    min_sep = 0.9 * sigma  
    R0 = sample_positions_min_sep(k1, N, dim, box_size, min_sep=min_sep)

    init_fn, step_fn = simulate.nve(
        energy_or_force_fn=lj_energy,
        shift_fn=shift,
        dt=dt,
        mass=mass
    )

    state = init_fn(key, R0, kT)

    @jax.jit
    def run_simulation(state):
        def body(st, _):
            st = step_fn(st)
            return st, (st.position, st.velocity)
        _, (pos, vel) = jax.lax.scan(body, state, None, length=num_steps)
        return pos, vel

    positions, velocities = run_simulation(state)
    KE = jax.vmap(lambda v: quantity.kinetic_energy(velocity=v, mass=mass), in_axes=0)(velocities)
    PE = jax.vmap(lj_energy, in_axes=0)(positions)
    total_energy = KE + PE


    jnp.savez(f'{out_dir}/{fname}', pos=positions, vel=velocities)

    x_axis = range(cfg.num_steps)


    # plt.plot(x_axis, total_energy, label="total energy")
    # plt.plot(x_axis, KE, label="kinetic energy")
    # plt.plot(x_axis, PE, label="potential energy")

    # plt.legend()
    # plt.show()
    if is_train:
        return f'{out_dir}/{cfg.train_fname}'
    else:
        return f'{out_dir}/{cfg.val_fname}'

