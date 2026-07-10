import gc
from time import perf_counter
import numpy as np
from jax import jit, block_until_ready
from solvers import *


def bench_basic(H0, H1, A, omega_d, to_jit=False, **_):
    H = qt.QobjEvo([H0.to_qutip(), [H1.to_qutip(), lambda t: A * np.cos(omega_d * t)]])

    t0 = perf_counter()
    out = floquet_basic(H, omega_d)
    t_solver = perf_counter() - t0

    q, m = post_process_qutip(*out, omega_d)
    return dict(t_solver=t_solver, t_total=t_solver, q=np.array(q), m=np.array(m))


def bench_dq_basic(H0, H1, A, omega_d, to_jit=False, **_):
    prop_fn = jit(propagator) if to_jit else propagator
    solver_fn = jit(floquet_dq_basic) if to_jit else floquet_dq_basic

    # warmup: warms up dynamiqs ODE JIT; also triggers if we have JIT compiled
    U = prop_fn(H0, H1, A, omega_d)
    block_until_ready(solver_fn(U))
    del U; gc.collect()

    t0 = perf_counter()
    U = prop_fn(H0, H1, A, omega_d)
    block_until_ready(U)
    t_prop = perf_counter() - t0

    t1 = perf_counter()
    out = solver_fn(U)
    block_until_ready(out)
    t_solver = perf_counter() - t1

    q, m = post_process(*out, omega_d)
    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver,
                q=np.array(q), m=np.array(m))


def bench_cayley(H0, H1, A, omega_d, cayley_phi=0, to_jit=False, **_):
    prop_fn = jit(propagator) if to_jit else propagator
    solver_fn = jit(floquet_cayley) if to_jit else floquet_cayley

    U = prop_fn(H0, H1, A, omega_d)
    block_until_ready(solver_fn(U, cayley_phi))
    del U; gc.collect()

    t0 = perf_counter()
    U = prop_fn(H0, H1, A, omega_d)
    block_until_ready(U)
    t_prop = perf_counter() - t0

    t1 = perf_counter()
    out = solver_fn(U, cayley_phi)
    block_until_ready(out)
    t_solver = perf_counter() - t1

    q, m = post_process(*out, omega_d)
    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver,
                q=np.array(q), m=np.array(m))


def bench_sambe_sparse(H0, H1, A, omega_d, sambe_copies=12, to_jit=False, **_):
    solver_fn = (jit(floquet_sambe, static_argnames=('N', 'dense'))
                 if to_jit else floquet_sambe)

    block_until_ready(solver_fn(H0, H1, A, omega_d, N=sambe_copies, dense=False))
    gc.collect()

    t0 = perf_counter()
    out = solver_fn(H0, H1, A, omega_d, N=sambe_copies, dense=False)
    block_until_ready(out)
    t_total = perf_counter() - t0

    q, m = post_process(*out, omega_d)
    return dict(t_total=t_total, q=np.array(q), m=np.array(m))


def bench_sambe_dense(H0, H1, A, omega_d, sambe_copies=12, to_jit=False, **_):
    solver_fn = (jit(floquet_sambe, static_argnames=('N', 'dense'))
                 if to_jit else floquet_sambe)

    block_until_ready(solver_fn(H0, H1, A, omega_d, N=sambe_copies, dense=True))
    gc.collect()

    t0 = perf_counter()
    out = solver_fn(H0, H1, A, omega_d, N=sambe_copies, dense=True)
    block_until_ready(out)
    t_total = perf_counter() - t0

    q, m = post_process(*out, omega_d)
    return dict(t_total=t_total, q=np.array(q), m=np.array(m))


####################################################################################################
BENCH_FNS = {
    'basic':        bench_basic,
    'dq_basic':     bench_dq_basic,
    'cayley':       bench_cayley,
    'sambe_sparse': bench_sambe_sparse,
    'sambe_dense':  bench_sambe_dense,
}
