import gc
from time import perf_counter
import numpy as np
from jax import jit, block_until_ready
from solvers import *


def bench_basic(H0, H1, A, omega_d, to_jit=False, **_):
    H = qt.QobjEvo([H0.to_qutip(), [H1.to_qutip(), lambda t: A * np.cos(omega_d * t)]])

    t0 = perf_counter()
    quasienergies, modes, U = floquet_basic(H, omega_d)
    t_solver = perf_counter() - t0

    q, m = post_process_qutip(quasienergies, modes, omega_d)
    nonunit_max = nonunitarity(U)
    return dict(t_solver=t_solver, t_total=t_solver, q=np.array(q), m=np.array(m),
                nonunit_max=nonunit_max)


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
    nonunit_max = nonunitarity(U)
    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver,
                q=np.array(q), m=np.array(m),
                nonunit_max=nonunit_max)

def bench_dq_basic_ip(H0, H1, A, omega_d, to_jit=False, **_):
    prop_fn = jit(ip_propagator) if to_jit else ip_propagator
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
    nonunit_max = nonunitarity(U)
    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver,
                q=np.array(q), m=np.array(m),
                nonunit_max=nonunit_max)


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
    nonunit_max = nonunitarity(U)
    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver,
                q=np.array(q), m=np.array(m),
                nonunit_max=nonunit_max)

def bench_cayley_ip(H0, H1, A, omega_d, cayley_phi=0, to_jit=False, **_):
    prop_fn = jit(ip_propagator) if to_jit else ip_propagator
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
    nonunit_max = nonunitarity(U)
    return dict(t_prop=t_prop, t_solver=t_solver, t_total=t_prop + t_solver,
                q=np.array(q), m=np.array(m),
                nonunit_max=nonunit_max)

####################################################################################################
BENCH_FNS = {
    'basic':        bench_basic,
    'dq_basic':     bench_dq_basic,
    'dq_basic_ip':  bench_dq_basic_ip,
    'cayley':       bench_cayley,
    'cayley_ip':    bench_cayley_ip,
}
