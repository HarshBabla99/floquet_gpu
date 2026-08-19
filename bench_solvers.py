import gc
from time import perf_counter
import numpy as np
from jax import jit, block_until_ready
from solvers import *
from utils import nonunitarity


def make_qutip_bench(tol):
    """QuTiP's FloquetBasis at integration tolerance `tol`.

    Deliberately left monolithic (FloquetBasis propagates and diagonalizes in one call).

    Used twice: at ref_rtol_atol as the reference, and at rtol_atol as a benchmarked
    solver on equal tolerance footing with the dynamiqs solvers.
    """
    def bench(H0, H1, A, omega_d, **_):
        H = qt.QobjEvo([H0.to_qutip(), [H1.to_qutip(), lambda t: A * np.cos(omega_d * t)]])

        t0 = perf_counter()
        quasienergies, modes, U = qt_floquet(H, omega_d, tol=tol)
        t_diag = perf_counter() - t0

        q, m = qt_post_process(None, modes, omega_d, quasienergies=quasienergies)
        return dict(t_prop=float('nan'), t_polar=float('nan'), t_diag=t_diag,
                    t_total=t_diag, q=np.array(q), m=np.array(m),
                    nonunit_max=nonunitarity(U))

    return bench


def make_bench(prop_fn, diag_fn, post_process_fn, polar_fn=None, 
               qutip=False, to_jit=True, uses_phi=False):
    """Build a bench function from a propagator, a diagonalizer, and a polar-projection flag.

    prop_fn  : func to form the single period propagator
    diag_fn  : func to diagonalize the propagator
    post_process_fn : func for postprocessing the results
    polar_fn : func to project the propagator onto the nearest unitary before diagonalizing
    qutip    : flag to convert arrays to Qobjs before benchmarking
    to_jit   : jit the propagator, projection and diagonalizer
    uses_phi : diag_fn takes the extra cayley_phi argument

    Timings are split into t_prop / t_polar / t_diag. 
    nonunit_max is measured on the matrix actually handed to the diagonalizer.
    """
    def bench(H0, H1, A, omega_d, cayley_phi=0.0, **_):
        prop = jit(prop_fn) if to_jit else prop_fn
        proj = (jit(polar_fn) if to_jit else polar_fn) if polar_fn else None
        diag = jit(diag_fn) if to_jit else diag_fn

        if qutip:
            H0 = H0.to_qutip()
            H1 = H1.to_qutip()

        def diagonalize(U):
            return diag(U, cayley_phi) if uses_phi else diag(U)

        # Always warm up: triggers JIT compilation
        if to_jit:
            U = prop(H0, H1, A, omega_d)
            if proj is not None:
                U = proj(U)
            block_until_ready(diagonalize(U))
            del U; gc.collect()

        # Propagator
        t0 = perf_counter()
        U = prop(H0, H1, A, omega_d)
        block_until_ready(U)
        t_prop = perf_counter() - t0

        # Polar projection
        t_polar = 0.0
        if proj is not None:
            t1 = perf_counter()
            U = proj(U)
            block_until_ready(U)
            t_polar = perf_counter() - t1

        # Measure nonunitariness of U (not timed)
        nonunit_max = nonunitarity(U)

        # Diagonalize
        t2 = perf_counter()
        out = diagonalize(U)
        block_until_ready(out)
        t_diag = perf_counter() - t2

        # Post process (not timed)
        q, m = post_process_fn(*out, omega_d)
        return dict(t_prop=t_prop, t_polar=t_polar, t_diag=t_diag,
                    t_total=t_prop + t_polar + t_diag,
                    q=np.array(q), m=np.array(m), nonunit_max=nonunit_max)

    return bench


####################################################################################################
# Solver registry: (frame) x (diagonalizer) x (polar projection), all jitted.

# Final registry
BENCH_FNS = {}
CPU_ONLY_SOLVERS = []

# 1. Reference solver
REFERENCE_SOLVER = 'ref'
BENCH_FNS[REFERENCE_SOLVER] = make_qutip_bench(ref_rtol_atol)
CPU_ONLY_SOLVERS.append(REFERENCE_SOLVER)

# 2. Qutip and dynamiqs solvers
PROP_FNS = {
    # name : (qutip function,  dynamiqs function)
    'lab': (qt_lab_propagator, dq_lab_propagator),
    'ip':  (qt_ip_propagator,  dq_ip_propagator),
}
DIAG_FNS = {
    # name    : (qutip function, dynamiqs function, uses cayley_phi)
    'basic':    (qt_basic,       dq_basic,          False),
    'cayley':   (qt_cayley,      dq_cayley,         True),
}
POLAR_PROJECT_FNS = [
    qt_polar_project, dq_polar_project
]
POST_PROCESS_FNS = [
    qt_post_process, dq_post_process
]

for _bidx, _backend in enumerate(['qt', 'dq']):
    for _frame, _prop in PROP_FNS.items():
        for _diag_name, _diag in DIAG_FNS.items():
            for _polar in (False, True):
                _name = f'{_backend}_{_frame}_{_diag_name}' + ('_polar' if _polar else '')
                _polar_fn = POLAR_PROJECT_FNS[_bidx] if _polar else None

                BENCH_FNS[_name] = make_bench(prop_fn=_prop[_bidx], 
                                              diag_fn=_diag[_bidx], 
                                              post_process_fn=POST_PROCESS_FNS[_bidx], 
                                              polar_fn=_polar_fn, 
                                              qutip=(_backend == 'qt'),
                                              to_jit=(_backend == 'dq'), 
                                              uses_phi=_diag[-1])
                if _backend == 'qt':
                    CPU_ONLY_SOLVERS.append(_name)
                    