import gc
from time import perf_counter
import numpy as np
from jax import jit, block_until_ready
from solvers import *

def bench_dq_basic(H0, H1, A, omega_d, **_):
    prop_fn = jit(propagator)
    solver_fn = jit(floquet_dq_basic)

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

    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver)


def bench_cayley(H0, H1, A, omega_d, cayley_phi=0, **_):
    prop_fn = jit(propagator)
    solver_fn = jit(floquet_cayley)

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

    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver)

####################################################################################################
BENCH_FNS = {
    'dq_basic':     bench_dq_basic,
    'cayley':       bench_cayley,
}
