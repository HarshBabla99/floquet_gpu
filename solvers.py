import jax
import jax.numpy as jnp
import dynamiqs as dq

dq.set_precision('double')

##
# Shared step: discretize one drive period into N piecewise-constant
# intervals (midpoint rule) and batch-exponentiate each interval's generator.
# Identical for both composition strategies below -- the comparison is only
# about how the resulting step propagators get combined.
##
def batched_step_propagators(H0, H1, A, omega_d, N):
    T = 2.0 * jnp.pi / omega_d
    dt = T / N
    t_mid = (jnp.arange(N) + 0.5) * dt

    H0j, H1j = H0.to_jax(), H1.to_jax()
    Hs = H0j[None] + (A * jnp.cos(omega_d * t_mid))[:, None, None] * H1j[None]

    return jax.scipy.linalg.expm(-1j * dt * Hs)  # (N, d, d)


##
# Composition strategies
##
def propagator_expm_scan(step_propagators):
    """Replicates dynamiqs's current `ExpmIntegrator`: combines the batched
    step propagators sequentially via `jax.lax.scan` (O(N) sequential depth).
    """
    d = step_propagators.shape[-1]
    y0 = jnp.eye(d, dtype=step_propagators.dtype)

    def step(carry, x):
        return x @ carry, None

    U, _ = jax.lax.scan(step, y0, step_propagators)
    return U


def propagator_expm_assoc(step_propagators):
    """Combines the batched step propagators via `jax.lax.associative_scan`
    (O(log N) sequential depth instead of O(N)).
    """
    combine = lambda a, b: b @ a  # a earlier, b later: compose as b . a
    return jax.lax.associative_scan(combine, step_propagators)[-1]
