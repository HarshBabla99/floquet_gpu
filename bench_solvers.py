import gc
import jax
import jax.numpy as jnp
from time import perf_counter
from jax import block_until_ready
from solvers import batched_step_propagators, propagator_expm_scan, propagator_expm_assoc


def _bench_expm(compose_fn, H0, H1, A, omega_d, expm_steps, jit):
    """Times the two phases shared by both composition strategies:
    - t_batch:   build the N piecewise-constant generators and batch-exponentiate them
    - t_compose: combine the N step propagators into the full-period propagator
    `compose_fn` is the only thing that differs between solvers (scan vs associative_scan).
    """
    batch_fn = (jax.jit(batched_step_propagators, static_argnames='N')
                if jit else batched_step_propagators)
    compose = jax.jit(compose_fn) if jit else compose_fn

    # warmup: triggers JIT compilation when jit=True
    sp = batch_fn(H0, H1, A, omega_d, expm_steps)
    block_until_ready(compose(sp))
    del sp; gc.collect()

    t0 = perf_counter()
    sp = batch_fn(H0, H1, A, omega_d, expm_steps)
    block_until_ready(sp)
    t_batch = perf_counter() - t0

    t1 = perf_counter()
    U = compose(sp)
    block_until_ready(U)
    t_compose = perf_counter() - t1

    return dict(t_batch=t_batch, t_compose=t_compose, t_total=t_batch + t_compose, U=U, sp=sp)


def bench_expm_scan(H0, H1, A, omega_d, expm_steps=32, jit=False, **_):
    m = _bench_expm(propagator_expm_scan, H0, H1, A, omega_d, expm_steps, jit)
    return dict(t_batch=m['t_batch'], t_compose=m['t_compose'], t_total=m['t_total'])


def bench_expm_assoc(H0, H1, A, omega_d, expm_steps=32, jit=False, **_):
    m = _bench_expm(propagator_expm_assoc, H0, H1, A, omega_d, expm_steps, jit)

    # untimed correctness check against the scan-based (current dynamiqs) composition
    U_ref = propagator_expm_scan(m['sp'])
    perr = float(jnp.max(jnp.abs(m['U'] - U_ref)))

    return dict(t_batch=m['t_batch'], t_compose=m['t_compose'], t_total=m['t_total'], perr=perr)


####################################################################################################
BENCH_FNS = {
    'expm_scan':  bench_expm_scan,
    'expm_assoc': bench_expm_assoc,
}
