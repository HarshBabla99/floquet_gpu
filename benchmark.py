import argparse
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import dynamiqs as dq

from bench_solvers import BENCH_FNS, REFERENCE_SOLVER
from utils import *

# !!!!!!!!! model !!!!!!!!!
# from models import random_hermitian_matrices as model 
from models import transmon_vary_resonator_dim as model

####################################################################################################

def run(solver, d, run_index, device, cayley_phi, output_path,
        ref_dir=None):

    # Start up
    bench_fn = BENCH_FNS[solver]
    is_gpu = device.startswith('gpu')

    mem_start = peak_rss_mb()
    gpu_mem_start = gpu_peak_mb(is_gpu)
    print(f'Starting [{device}/{solver} run={run_index} d={d}] \t peak RSS={mem_start:.1f} MB', flush=True)

    # Define the problem
    # H0, H1, A, omega_d = random_hermitian_matrices(run_index, d)
    H0, H1, A, omega_d = model(run_index, d)

    # Characterise the model itself, so downstream analysis never has to rebuild it.
    # ||H0||*T is the phase accumulated over one drive period; it, not the Hilbert
    # dimension, is what sets the achievable accuracy at a given ODE tolerance.
    # H1 is deliberately not measured: it is dense, so its norm costs an O(N^3) eigvalsh,
    # and A*||H1|| is small next to ||H0|| anyway.
    h0_norm = hamiltonian_norm(H0)
    period = 2.0 * np.pi / omega_d
    model_info = dict(
        h0_norm=h0_norm,
        omega_d=omega_d, T=period, h0_normT=h0_norm * period,
        A_abs=float(np.abs(A)),   # A is complex-typed; its sign/phase is a gauge choice
    )

    # Reference solution. It may be missing (e.g. the reference job timed out at this
    # dim); in that case still record timing/memory, with NaN errors.
    reference = None
    if solver != REFERENCE_SOLVER:
        _ref_dir = ref_dir if ref_dir is not None else os.path.dirname(output_path)
        ref_path = os.path.join(_ref_dir, f'{REFERENCE_SOLVER}_d{d}_run{run_index}.npy')
        try:
            reference = load_rows(ref_path)[0]
        except (OSError, IndexError) as e:
            print(f'WARNING: no reference at {ref_path} ({type(e).__name__}); '
                  f'recording timings with qerr/merr=NaN', flush=True)

    # Run the benchmark
    metrics = bench_fn(H0, H1, A, omega_d, cayley_phi=cayley_phi)

    # Add params, runtimes, and errs (if not reference) to the row
    if solver == REFERENCE_SOLVER:
        row = dict(solver=solver, device=device, run_index=run_index, d=d,
                   **model_info, **metrics)
    else:
        if reference is None:
            qerr = merr = float('nan')
        else:
            # Compare quasienergies/modes against the reference, pairing modes by maximum
            # overlap (not sorted by quasienergy position)
            overlap = np.abs(reference['m'].conj() @ metrics['m'].T)**2
            ref_idx, test_idx = linear_sum_assignment(-overlap)
            merr = float(np.max(1.0 - overlap[ref_idx, test_idx]))
            qerr = float(np.max(np.abs(reference['q'][ref_idx] - metrics['q'][test_idx])))

        row = dict(
            solver=solver, device=device, run_index=run_index, d=d, **model_info,
            t_total=metrics['t_total'], t_prop=metrics['t_prop'],
            t_polar=metrics['t_polar'], t_diag=metrics['t_diag'],
            qerr=qerr, merr=merr, ref_missing=reference is None,
            nonunit_max=metrics['nonunit_max'],
        )

    # CPU memory
    mem_final = peak_rss_mb()
    row['mem_total'] = mem_final - mem_start

    # GPU memory
    if is_gpu:
        gpu_mem_final = gpu_peak_mb(is_gpu)
        row['mem_gpu'] = (gpu_mem_final - gpu_mem_start
                          if gpu_mem_final is not None and gpu_mem_start is not None
                          else None)

    # Save
    with open(output_path, 'ab') as f:
        np.save(f, row)

    # Final print
    final_str = f'Finished [{device}/{solver} run={run_index} d={d}] \t peak RSS={mem_final:.1f} MB'
    final_str += f'\tt_total={metrics["t_total"]:.3f}s'
    final_str += f'\tqerr={row["qerr"]:.3e} \t merr={row["merr"]:.3e}' if 'qerr' in row else ''
    final_str += f'\tnonunit_max={row["nonunit_max"]:.3e}'
    print(final_str, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver',    required=True, choices=list(BENCH_FNS))
    parser.add_argument('--dim',       type=int, required=True)
    parser.add_argument('--run-index', type=int, required=True)
    parser.add_argument('--device',    default='cpu',
                        help='Device identifier passed to dq.set_device; also used as output subdirectory')
    parser.add_argument('--ref-dir', default=None,
                        help=f'Directory containing {REFERENCE_SOLVER}_*.npy reference files; '
                             'defaults to the same directory as the output file')
    args = parser.parse_args()

    cayley_phi = 0.

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
        output_path=output_path,
        ref_dir=args.ref_dir)