import gc
from time import perf_counter
import numpy as np
from jax import jit, block_until_ready
from solvers import *


def bench_ref(H0, H1, A, omega_d, **_):
    """Reference: QuTiP's FloquetBasis at ref_rtol_atol.

    Deliberately left monolithic (FloquetBasis propagates and diagonalizes in one call),
    so the reference stays the stock QuTiP entry point rather than a variant of our own.
    """
    H = qt.QobjEvo([H0.to_qutip(), [H1.to_qutip(), lambda t: A * np.cos(omega_d * t)]])

    t0 = perf_counter()
    quasienergies, modes, U = floquet_basic(H, omega_d)
    t_diag = perf_counter() - t0

    q, m = post_process_qutip(quasienergies, modes, omega_d)
    return dict(t_prop=float('nan'), t_polar=float('nan'), t_diag=t_diag,
                t_total=t_diag, q=np.array(q), m=np.array(m),
                nonunit_max=nonunitarity(U))


def make_bench(prop_fn, diag_fn, polar=False, to_jit=True, uses_phi=False):
    """Build a bench function from a propagator, a diagonalizer, and a polar-projection flag.

    prop_fn  : `propagator` (lab frame) or `ip_propagator` (interaction picture)
    diag_fn  : `floquet_dq_basic` (eig) or `floquet_cayley`
    polar    : project the propagator onto the nearest unitary before diagonalizing
    to_jit   : jit the propagator, projection and diagonalizer
    uses_phi : diag_fn takes the extra cayley_phi argument

    Timings are split into t_prop / t_polar / t_diag. nonunit_max is measured on the
    matrix actually handed to the diagonalizer, so with polar=True it reports the
    projection's residual (~1e-15) rather than the raw integration error -- compare
    against the matching polar=False solver to see what the projection removed.
    """
    def bench(H0, H1, A, omega_d, cayley_phi=0.0, **_):
        prop = jit(prop_fn) if to_jit else prop_fn
        proj = (jit(polar_project) if to_jit else polar_project) if polar else None
        diag = jit(diag_fn) if to_jit else diag_fn

        def diagonalize(U):
            return diag(U, cayley_phi) if uses_phi else diag(U)

        # Always warm up: triggers JIT compilation, and dynamiqs' internals need it
        # even when to_jit is False.
        U = prop(H0, H1, A, omega_d)
        if proj is not None:
            U = proj(U)
        block_until_ready(diagonalize(U))
        del U; gc.collect()

        t0 = perf_counter()
        U = prop(H0, H1, A, omega_d)
        block_until_ready(U)
        t_prop = perf_counter() - t0

        t_polar = 0.0
        if proj is not None:
            t1 = perf_counter()
            U = proj(U)
            block_until_ready(U)
            t_polar = perf_counter() - t1

        # Measured on the diagonalizer's input, outside every timed region.
        nonunit_max = nonunitarity(U)

        t2 = perf_counter()
        out = diagonalize(U)
        block_until_ready(out)
        t_diag = perf_counter() - t2

        q, m = post_process(*out, omega_d)
        return dict(t_prop=t_prop, t_polar=t_polar, t_diag=t_diag,
                    t_total=t_prop + t_polar + t_diag,
                    q=np.array(q), m=np.array(m), nonunit_max=nonunit_max)

    return bench


####################################################################################################
# Solver registry: (frame) x (diagonalizer) x (polar projection), all jitted.

REFERENCE_SOLVER = 'ref'

PROP_FNS = {
    'lab': propagator,
    'ip':  ip_propagator,
}
DIAG_FNS = {
    # name    : (function,           uses cayley_phi)
    'basic':    (floquet_dq_basic,   False),
    'cayley':   (floquet_cayley,     True),
}

BENCH_FNS = {REFERENCE_SOLVER: bench_ref}
for _frame, _prop in PROP_FNS.items():
    for _diag_name, (_diag, _uses_phi) in DIAG_FNS.items():
        for _polar in (False, True):
            _name = f'{_frame}_{_diag_name}' + ('_polar' if _polar else '')
            BENCH_FNS[_name] = make_bench(_prop, _diag, polar=_polar,
                                          to_jit=True, uses_phi=_uses_phi)
