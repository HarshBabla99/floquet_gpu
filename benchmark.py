import argparse
import os
import resource
import sys
import numpy as np
import scqubits as scq

from jax import devices as jdevices
from jax.numpy.linalg import norm
import jax.random as jrand
import dynamiqs as dq

from bench_solvers import BENCH_FNS

ALL_SOLVERS = ['basic'] + [k + '_jit' for k in BENCH_FNS if k != 'basic'] 
#+ list(BENCH_FNS)

####################################################################################################
# Floquet models

def random_hermitian(shape, key):
    """Generate a random Hermitian matrix with unit norm."""
    H = dq.random.herm(key, shape)
    _norm = norm(H.to_jax(), 2)
    return H / _norm

def random_hermitian_matrices(run_index, d):
    k0, k1, k2, k3 = jrand.split(jrand.fold_in(jrand.key(run_index), d), num=4)
    h0_scale=100
    H0 = h0_scale*random_hermitian((d, d), k0)
    H1 = random_hermitian((d, d), k1)
    A = float(jrand.uniform(k2, minval=0.5, maxval=1.5))
    omega_d = float(jrand.uniform(k3, minval=1.0, maxval=5.0))

    return H0, H1, A, omega_d

def get_transmon(omega_p=1, zeta=0.25, truncated_dim=20):
    """Generate a deterministic transmon. zeta=0.25 results in EJ/EC approx 100"""

    scq_qubit_params = {
        'EC' : omega_p * zeta/8,
        'EJ' : omega_p / zeta,
        'ncut': 41,
        'ng' : 0.0,
        'truncated_dim' : truncated_dim,
    }
    tmon = scq.Transmon(**scq_qubit_params)
    return tmon

def get_floquet_params_from_scq(hilbert_space, tmon, k2, k3):
    # Diagonalize
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0]
    evals = evals - evals[0]

    # Define the operators
    H0 = 2.0 * np.pi * dq.sparsedia_from_dict({0:evals})
    
    H1_np = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)
    drive_matelem = H1_np[0, 1]
    H1 = dq.asqarray(H1_np)

    # Drive params
    xi_sq = float(jrand.uniform(k2, minval=0.5, maxval=3.0))
    omega_d = float(jrand.uniform(k3, minval=1.2, maxval=7.0))
    A = 0.5 * np.sqrt(xi_sq) * (1 - omega_d**2)/(drive_matelem * omega_d)

    return H0, H1, A, omega_d

def transmon_vary_resonator_dim(run_index, d):
    k0, k1, k2, k3 = jrand.split(jrand.fold_in(jrand.key(run_index), d), num=4)

    # Deterministic transmon
    tmon = get_transmon()

    # random resonator and coupling
    # resonator freq = 1.2 to 3.0 times qubit frequency
    # g = (1/20) to (1/30) times qubit freq
    omega_r = float(jrand.uniform(k0, minval=1.2, maxval=3.0))
    g = float(jrand.uniform(k1, minval=(1/30), maxval=(1/20)))

    res = scq.Oscillator(E_osc = omega_r, truncated_dim=d)

    hilbert_space = scq.HilbertSpace([tmon, res])
    hilbert_space.add_interaction(
        g_strength = g,
        op1 = tmon.n_operator,
        op2 = res.annihilation_operator,
        add_hc = True
    )

    return get_floquet_params_from_scq(hilbert_space, tmon, k2, k3)

####################################################################################################

def _peak_rss_mb():
    """Peak memory of this process (on host), in MB."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux, bytes on macOS.
    return ru_maxrss / (1024 if sys.platform != 'darwin' else 1024 * 1024)

def _gpu_peak_mb(is_gpu):
    """Peak memory of this process (on the GPU), in MB. None if not supported"""
    if not is_gpu:
        return None
    stats = jdevices()[0].memory_stats()
    if not stats:
        return None
    return stats['peak_bytes_in_use'] / 1024**2

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

def run(solver, d, run_index, device, cayley_phi, sambe_copies, output_path,
        basic_dir=None):
    to_jit = solver.endswith('_jit')
    base_solver = solver.removesuffix('_jit') if to_jit else solver
    bench_fn = BENCH_FNS[base_solver]
    is_gpu = device.startswith('gpu')

    mem_start = _peak_rss_mb()
    gpu_mem_start = _gpu_peak_mb(is_gpu)
    print(f'Starting [{device}/{solver} run={run_index} d={d}] \t peak RSS={mem_start:.1f} MB', flush=True)

    # Define the problem
    # H0, H1, A, omega_d = random_hermitian_matrices(run_index, d)
    H0, H1, A, omega_d = transmon_vary_resonator_dim(run_index, d)

    # Reference solution from the qutip 'basic' solver. It may be missing (e.g. the basic
    # job timed out at this dim); in that case still record timing/memory, with NaN errors.
    basic_ref = None
    if base_solver != 'basic':
        _basic_dir = basic_dir if basic_dir is not None else os.path.dirname(output_path)
        basic_path = os.path.join(_basic_dir, f'basic_d{d}_run{run_index}.npy')
        try:
            basic_ref = load_rows(basic_path)[0]
        except (OSError, IndexError) as e:
            print(f'WARNING: no basic reference at {basic_path} ({type(e).__name__}); '
                  f'recording timings with qerr/merr=NaN', flush=True)

    metrics = bench_fn(H0, H1, A, omega_d,
                       cayley_phi=cayley_phi, sambe_copies=sambe_copies, to_jit=to_jit)

    if base_solver == 'basic':
        row = dict(solver=solver, device=device, run_index=run_index, d=d, **metrics)
    else:
        if basic_ref is None:
            qerr = merr = float('nan')
        else:
            innerp = np.sum(basic_ref['m'].conj() * metrics['m'], axis=1)
            qerr = float(np.max(np.abs(basic_ref['q'] - metrics['q'])))
            merr = float(np.max(1.0 - np.abs(innerp)**2))

        row = dict(
            solver=solver, device=device, run_index=run_index, d=d,
            t_total=metrics['t_total'],
            qerr=qerr, merr=merr, ref_missing=basic_ref is None,
        )
        if 't_prop' in metrics:
            row['t_prop'] = metrics['t_prop']
            row['t_solver'] = metrics['t_solver']

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
    final_str += f'\tqerr={row["qerr"]:.3e} \t merr={row["merr"]:.3e}' if 'qerr' in row else ''
    print(final_str, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver',    required=True, choices=ALL_SOLVERS)
    parser.add_argument('--dim',       type=int, required=True)
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--device',    default='cpu',
                        help='Device identifier passed to dq.set_device; also used as output subdirectory')
    parser.add_argument('--basic-dir', default=None,
                        help='Directory containing basic reference .npy files; '
                             'defaults to the same directory as the output file')
    args = parser.parse_args()

    cayley_phi = 0.
    sambe_copies = 12

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
        cayley_phi=cayley_phi,
        sambe_copies=sambe_copies,
        output_path=output_path,
        basic_dir=args.basic_dir)