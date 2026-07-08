import argparse
import os
import resource
import sys
import numpy as np

import jax
from jax.numpy.linalg import norm
import jax.random as jrand
import dynamiqs as dq

from bench_solvers import BENCH_FNS

ALL_SOLVERS = list(BENCH_FNS) + [k + '_jit' for k in BENCH_FNS]

def _peak_rss_mb():
    """Peak memory of this process (on host), in MB."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux, bytes on macOS.
    return ru_maxrss / (1024 if sys.platform != 'darwin' else 1024 * 1024)

def _gpu_peak_mb(is_gpu):
    """Peak memory of this process (on the GPU), in MB. None if not supported"""
    if not is_gpu:
        return None
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    return stats['peak_bytes_in_use'] / 1024**2

def random_hermitian(shape, key):
    """Generate a random Hermitian matrix with unit norm."""
    H = dq.random.herm(key, shape)
    _norm = norm(H.to_jax(), 2)
    return H / _norm

def load_rows(path):
    """Read all rows sequentially from a .npy file written in append mode."""
    rows = []
    with open(path, 'rb') as f:
        while True:
            try:
                rows.append(np.load(f, allow_pickle=True).item())
            except (EOFError, ValueError):
                break
    return rows

def run(solver, d, run_index, device, A, omega_d, expm_steps, output_path):
    is_jit = solver.endswith('_jit')
    base_solver = solver.removesuffix('_jit') if is_jit else solver
    bench_fn = BENCH_FNS[base_solver]
    is_gpu = device.startswith('gpu')

    mem_start = _peak_rss_mb()
    gpu_mem_start = _gpu_peak_mb(is_gpu)
    print(f'Starting [{device}/{solver} run={run_index} d={d}] \t peak RSS={mem_start:.1f} MB', flush=True)

    k0, k1 = jrand.split(jrand.fold_in(jrand.key(run_index), d))
    H0 = random_hermitian((d, d), k0)
    H1 = random_hermitian((d, d), k1)

    metrics = bench_fn(H0, H1, A, omega_d, expm_steps=expm_steps, jit=is_jit)
    row = dict(solver=solver, device=device, run_index=run_index, d=d, **metrics)

    mem_final = _peak_rss_mb()
    row['mem_total'] = mem_final - mem_start

    if is_gpu:
        gpu_mem_final = _gpu_peak_mb(is_gpu)
        row['mem_gpu'] = (gpu_mem_final - gpu_mem_start
                          if gpu_mem_final is not None and gpu_mem_start is not None
                          else None)

    with open(output_path, 'ab') as f:
        np.save(f, row)

    final_str = f'Finished [{device}/{solver} run={run_index} d={d}] \t peak RSS={mem_final:.1f} MB \t'
    final_str += f't_total={metrics["t_total"]:.3f}s'
    final_str += f'\tperr={row["perr"]:.3e}' if 'perr' in row else ''
    print(final_str, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver',    required=True, choices=ALL_SOLVERS)
    parser.add_argument('--dim',       type=int, required=True)
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--device',    default='cpu',
                        help='Device identifier passed to dq.set_device; also used as output subdirectory')
    parser.add_argument('--expm-steps', type=int, default=32,
                        help='Number N of piecewise-constant sub-intervals used to '
                             'discretize one drive period before batched matrix '
                             'exponentiation (shared by both composition strategies).')
    args = parser.parse_args()

    A = 1.0
    omega_d = 2.0

    out_dir = os.path.join('out', args.device)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'{args.solver}_d{args.dim}_run{args.run_index}.npy')

    jax_device = 'gpu' if args.device.startswith('gpu') else 'cpu'
    print(f'Setting device to: {jax_device}', flush=True)
    dq.set_device(jax_device)

    run(solver=args.solver,
        d=args.dim,
        run_index=args.run_index,
        device=args.device,
        A=A, omega_d=omega_d,
        expm_steps=args.expm_steps,
        output_path=output_path)