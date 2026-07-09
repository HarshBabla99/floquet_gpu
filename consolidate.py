import argparse
import glob
import os
import numpy as np
from benchmark import load_rows, ALL_SOLVERS


def load_solver_runs(solver, out_dir='out'):
    """Return {d: [row, row, ...]} aggregated across all run files for a solver.

    Large array-valued fields (e.g. the raw propagator `U` kept around for
    other solvers to use as a reference) are dropped here -- they belong in
    the per-run raw files, not in the aggregated summary.
    """
    by_dim = {}
    for path in sorted(glob.glob(f'{out_dir}/{solver}_d*_run*.npy')):
        for row in load_rows(path):
            row = {k: v for k, v in row.items() if not hasattr(v, 'shape')}
            by_dim.setdefault(row['d'], []).append(row)
    return by_dim

def consolidate(out_dir, output_path, solvers=None):

    # Default list of solvers in benchmark.py
    target = solvers if solvers is not None else ALL_SOLVERS

    # Load existing consolidated file, if exists
    data = np.load(output_path, allow_pickle=True).item() if os.path.exists(output_path) else {}

    # Load for each solver
    for solver in target:
        runs = load_solver_runs(solver, out_dir=out_dir)
        data[solver] = runs
        n_rows = sum(len(rows) for rows in runs.values())
        print(f'  {solver:15s}: {len(runs):2d} dims, {n_rows:4d} runs')

    np.save(output_path, data)
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='out',
                        help='Directory containing per-run .npy files')
    parser.add_argument('--output',  default=None,
                        help='Output path (default: <out-dir>/_consolidated.npy)')
    parser.add_argument('--solvers', nargs='+', default=None,
                        help='Solvers to include (default: all). '
                             'e.g. --solvers basic  or  --solvers dq_basic cayley sambe_sparse sambe_dense')
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.out_dir, '_consolidated.npy')
    consolidate(args.out_dir, output_path, solvers=args.solvers)
