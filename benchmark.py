import argparse
import os
import resource
import sys
import time
from numpy import save

from jax.numpy.linalg import norm
from jax import block_until_ready
import jax.random as jrand
from jax import devices as jdevices
from jax import profiler as jprofiler
import dynamiqs as dq

from solvers import SOLVERS
from profiling import summarize_trace, kernel_summary, phase_map

# Env vars that change what the compiled program does or how it executes, and so
# must be recorded alongside every measurement. Snapshotted from the environment
# actually in effect rather than from CLI arguments, so the record cannot drift
# from what really ran.
_ENV_PREFIXES = ('XLA_', 'CUBLAS_', 'JAX_', 'CUDA_', 'NVIDIA_', 'LIBCUBLAS_')

def _env_snapshot():
    """Provenance: the execution-affecting environment this run saw."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(_ENV_PREFIXES)}

def _peak_rss_mb():
    """Peak memory of this process (on host), in MB."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux, bytes on macOS.
    return ru_maxrss / (1024 if sys.platform != 'darwin' else 1024**2)

def _gpu_peak_mb(is_gpu):
    """Peak memory of this process (on the GPU), in MB. None if not supported"""
    if not is_gpu:
        return None
    stats = jdevices()[0].memory_stats()
    if not stats:
        return None
    return stats['peak_bytes_in_use'] / 1024**2

def random_hermitian(shape, key):
    """Generate a random Hermitian matrix with unit norm."""
    H = dq.random.herm(key, shape)
    _norm = norm(H.to_jax(), 2)
    return H / _norm

def run(solver, d, run_index, device, tag, A, omega_d, cayley_phi, output_path, trace_root):

    # Get the solver and device
    solver_fn = SOLVERS[solver]
    is_gpu = device.startswith('gpu')

    mem_start = _peak_rss_mb()
    gpu_mem_start = _gpu_peak_mb(is_gpu)
    print(f'Starting [{device}/{solver} run={run_index} d={d}] \t peak RSS={mem_start:.1f} MB', flush=True)

    k0, k1 = jrand.split(jrand.fold_in(jrand.key(run_index), d))
    H0 = random_hermitian((d, d), k0)
    H1 = random_hermitian((d, d), k1)

    # Run warmup
    _ = block_until_ready(solver_fn(H0, H1, A, omega_d, cayley_phi=cayley_phi))

    # Read the compiler's own propagator/diagonalisation split (solvers.py marks
    # the two stages with jax.named_scope). The executable is already in JAX's
    # compilation cache from the warmup, so this costs nothing to re-lower.
    try:
        hlo = solver_fn.lower(H0, H1, A, omega_d,
                              cayley_phi=cayley_phi).compile().as_text()
        pmap = phase_map(hlo)
    except Exception as exc:                                   # noqa: BLE001
        print(f'  (phase map unavailable: {exc!r})', flush=True)
        pmap = None

    # Timed steady-state run. Deliberately NOT profiled: starting/stopping the
    # profiler costs ~100 ms on GPU and ~0.5 s on CPU, which swamps the solver
    # itself at small d. This is the wall-clock number to trust.
    t0 = time.perf_counter()
    _ = block_until_ready(solver_fn(H0, H1, A, omega_d, cayley_phi=cayley_phi))
    t1 = time.perf_counter()

    # Read the memory high-water marks here, before any profiler machinery runs.
    # ru_maxrss is a high-water mark, so parsing the (large) xplane protobuf below
    # would otherwise be charged to the solver.
    mem_final = _peak_rss_mb()
    gpu_mem_final = _gpu_peak_mb(is_gpu)

    # Separate traced run, purely to capture the trace. Its wall time is recorded
    # as t_total_traced so profiler overhead stays visible, but it is not t_total.
    trace_filename = os.path.join(trace_root, f'{solver}_d{d}_run{run_index}')
    os.makedirs(trace_filename, exist_ok=True)

    t2 = time.perf_counter()
    with jprofiler.trace(trace_filename):
        _ = block_until_ready(
            solver_fn(H0, H1, A, omega_d, cayley_phi=cayley_phi)
        )
    t3 = time.perf_counter()
    jprofiler.save_device_memory_profile(os.path.join(trace_filename, 'memory.prof'))

    # Save profiling results
    row = dict(
        solver=solver, device=device, run_index=run_index, d=d,
        t_total=t1 - t0,
        t_total_traced=t3 - t2,
        trace_dir=trace_filename,
        # Provenance: which configuration produced this row. Device-time
        # attribution is NOT comparable across variants - a while-loop captured
        # as a CUDA graph reports launch cost, not kernel cost.
        tag=tag,
        xla_flags=os.environ.get('XLA_FLAGS', ''),
        env=_env_snapshot(),
        # Condensed kernel-level timeline, so the analysis never has to re-read
        # the .xplane.pb files and they can be deleted after the run.
        kernels=kernel_summary(trace_filename, pmap=pmap),
        **summarize_trace(trace_filename, pmap=pmap),
    )

    # Memory usage
    row['mem_total'] = mem_final - mem_start

    if is_gpu:
        row['mem_gpu'] = (gpu_mem_final - gpu_mem_start
                          if gpu_mem_final is not None and gpu_mem_start is not None
                          else None)

    with open(output_path, 'ab') as f:
        save(f, row)

    # Print results
    top_op, top_ns = row['top_ops'][0] if row['top_ops'] else ('n/a', 0)
    final_str = f'Finished [{device}/{solver} run={run_index} d={d}] \t'
    final_str += f'peak RSS={mem_final:.1f} MB \t'
    final_str += f't_total={row["t_total"]:.3f}s \t'
    final_str += f'traced={row["t_total_traced"]:.3f}s \t'
    final_str += f'top_op={top_op} ({top_ns / 1e6:.3f}ms)'
    print(final_str, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver',    required=True, choices=list(SOLVERS))
    parser.add_argument('--dim',       type=int, required=True)
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--device',    default='cpu',
                        help='Device identifier passed to dq.set_device; also used as output subdirectory')
    parser.add_argument('--tag',       default='',
                        help='Label for the XLA configuration of this run (e.g. cmdbuf, '
                             'nocmdbuf). Appended to the output subdirectory and recorded '
                             'in every row, so variants never share an output file.')
    args = parser.parse_args()

    A = 1.0
    omega_d = 2.0
    cayley_phi = 0.

    out_dir = os.path.join('out', f'{args.device}_{args.tag}' if args.tag else args.device)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'{args.solver}_d{args.dim}_run{args.run_index}.npy')
    trace_root = os.path.join(out_dir, 'traces')
    os.makedirs(trace_root, exist_ok=True)

    jax_device = 'gpu' if args.device.startswith('gpu') else 'cpu'
    print(f'Setting device to: {jax_device}', flush=True)
    dq.set_device(jax_device)

    run(solver=args.solver,
        d=args.dim,
        run_index=args.run_index,
        device=args.device,
        tag=args.tag,
        A=A, omega_d=omega_d,
        cayley_phi=cayley_phi,
        output_path=output_path,
        trace_root=trace_root)