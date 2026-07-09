import gc
import jax
import jax.numpy as jnp
from time import perf_counter
from jax import block_until_ready
from solvers import (batched_step_propagators, propagator_expm_scan,
                      propagator_expm_assoc, propagator_tsit5)


def bench_dq_basic(H0, H1, A, omega_d, jit=False, **_):
    prop_fn = jax.jit(propagator_tsit5) if jit else propagator_tsit5
    
    # unconditional warmup
    block_until_ready(prop_fn(H0, H1, A, omega_d))

    t0 = perf_counter()
    U = prop_fn(H0, H1, A, omega_d)
    block_until_ready(U)
    t_prop = perf_counter() - t0

    return dict(t_prop=t_prop, t_total=t_prop, U=U)


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

    return dict(t_batch=t_batch, t_compose=t_compose, t_total=t_batch + t_compose, U=U)


def _bench_expm_row(compose_fn, H0, H1, A, omega_d, expm_steps, jit, ref_U):
    m = _bench_expm(compose_fn, H0, H1, A, omega_d, expm_steps, jit)
    row = dict(t_batch=m['t_batch'], t_compose=m['t_compose'], t_total=m['t_total'])
    if ref_U is not None:
        row['perr'] = float(jnp.max(jnp.abs(m['U'] - ref_U)))
    return row


def bench_expm_scan(H0, H1, A, omega_d, expm_steps=32, jit=False, ref_U=None, **_):
    return _bench_expm_row(propagator_expm_scan, H0, H1, A, omega_d, expm_steps, jit, ref_U)


def bench_expm_assoc(H0, H1, A, omega_d, expm_steps=32, jit=False, ref_U=None, **_):
    return _bench_expm_row(propagator_expm_assoc, H0, H1, A, omega_d, expm_steps, jit, ref_U)


####################################################################################################
BENCH_FNS = {
    'dq_basic':   bench_dq_basic,
    'expm_scan':  bench_expm_scan,
    'expm_assoc': bench_expm_assoc,
}
