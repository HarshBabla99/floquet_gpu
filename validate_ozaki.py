"""Standalone correctness + timing check for ozaki.py, decoupled from the
Floquet solver pipeline. Compares against plain NumPy / SciPy.

NOTE: without a CUDA device, this runs on CPU/XLA:CPU. INT8 dot_general has
no tensor-core path to exploit there, so timings here are NOT evidence of
GPU speedup -- they only validate correctness and show host-side overhead.
Real performance numbers need to come from the SLURM GPU nodes.
"""
from time import perf_counter

import numpy as np
import jax

from ozaki import ozaki_matmul, ozaki_solve

jax.config.update('jax_enable_x64', True)

rng = np.random.default_rng(0)


def rel_err(x, y):
    return np.max(np.abs(x - y)) / np.max(np.abs(y))


def random_hermitian(d):
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = 0.5 * (X + X.conj().T)
    return H / np.linalg.norm(H, 2)


def random_unitary(d):
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(X)
    return Q


def timeit(fn, *args, reps=3):
    fn(*args)  # warmup / compile
    jax.block_until_ready(fn(*args))
    t0 = perf_counter()
    for _ in range(reps):
        out = fn(*args)
        jax.block_until_ready(out)
    return (perf_counter() - t0) / reps, out


def bench_matmul():
    print('\n=== ozaki_matmul vs numpy (complex128 Hermitian pairs) ===')
    print(f'{"d":>5} {"n_slices":>9} {"rel_err":>12} {"ozaki_ms":>10} {"numpy_ms":>10}')
    for d in [16, 32, 64, 128, 256, 512]:
        H0 = random_hermitian(d)
        H1 = random_hermitian(d)
        truth = H0 @ H1

        t_np0 = perf_counter()
        for _ in range(3):
            H0 @ H1
        t_np = (perf_counter() - t_np0) / 3

        for n_slices in [2, 4, 6, 8]:
            t_oz, C = timeit(lambda A, B: ozaki_matmul(A, B, n_slices), H0, H1)
            err = rel_err(np.array(C), truth)
            print(f'{d:>5} {n_slices:>9} {err:>12.3e} {t_oz*1e3:>10.3f} {t_np*1e3:>10.3f}')


def bench_solve():
    print('\n=== ozaki_solve vs scipy.linalg.solve (Cayley-like I + e^{i phi} U) ===')
    print(f'{"d":>5} {"n_slices":>9} {"max_iters":>10} {"rel_err":>12} {"ozaki_ms":>10} {"scipy_ms":>10}')
    import scipy.linalg as sla
    for d in [16, 32, 64, 128, 256, 512]:
        U = random_unitary(d)
        phi = 0.3  # avoid exact eigenvalue -1 singularity, as in floquet_cayley
        W = np.exp(1j * phi) * U
        A = np.eye(d) + W
        B = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))

        truth = sla.solve(A, B)
        t_sp0 = perf_counter()
        for _ in range(3):
            sla.solve(A, B)
        t_sp = (perf_counter() - t_sp0) / 3

        for n_slices, max_iters in [(4, 1), (4, 3), (6, 3), (8, 5)]:
            t_oz, X = timeit(
                lambda A_, B_: ozaki_solve(A_, B_, n_slices, max_iters), A, B)
            err = rel_err(np.array(X), truth)
            print(f'{d:>5} {n_slices:>9} {max_iters:>10} {err:>12.3e} '
                  f'{t_oz*1e3:>10.3f} {t_sp*1e3:>10.3f}')


if __name__ == '__main__':
    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}')
    bench_matmul()
    bench_solve()
