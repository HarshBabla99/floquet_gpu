import argparse
import glob
import os
import numpy as np
from solvers import SOLVERS

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

def load_solver_runs(solver, out_dir='out'):
    """Return {d: [row, row, ...]} aggregated across all run files for a solver."""
    by_dim = {}
    for path in sorted(glob.glob(f'{out_dir}/{solver}_d*_run*.npy')):
        for row in load_rows(path):
            by_dim.setdefault(row['d'], []).append(row)
    return by_dim

def _row_key(row):
    """Identity of a single benchmark run within one solver."""
    return (row.get('d'), row.get('run_index'))


def consolidate(out_dir, output_path, solvers=None):

    # Default list of solvers in benchmark.py
    target = solvers if solvers is not None else SOLVERS

    # Load existing consolidated file, if exists
    data = np.load(output_path, allow_pickle=True).item() if os.path.exists(output_path) else {}

    # Load for each solver
    for solver in target:
        runs = load_solver_runs(solver, out_dir=out_dir)
        n_rows = sum(len(rows) for rows in runs.values())

        # Finding nothing means the raw per-run files are missing - either the
        # jobs have not run, or they were deleted after a previous consolidation.
        # Overwriting in that case would silently destroy results that are not
        # reproducible without re-running the whole array, so keep what we have.
        if n_rows == 0:
            if solver in data:
                kept = sum(len(rows) for rows in data[solver].values())
                print(f'  {solver:15s}: no raw files found - KEEPING existing '
                      f'{kept} run(s) already in {os.path.basename(output_path)}')
            else:
                print(f'  {solver:15s}: no raw files found, nothing to add')
            continue

        # Merge rather than replace, keyed on (dim, run_index). This makes the
        # raw per-run files disposable: once a variant is consolidated its
        # directory can be deleted, and a later batch of extra runs tops the
        # file up instead of replacing it with only the new rows. Re-running an
        # existing run_index overwrites that one row.
        existing = data.get(solver, {})
        merged = {dim: {_row_key(r): r for r in rows} for dim, rows in existing.items()}
        added = replaced = 0
        for dim, rows in runs.items():
            slot = merged.setdefault(dim, {})
            for r in rows:
                key = _row_key(r)
                if key in slot:
                    replaced += 1
                else:
                    added += 1
                slot[key] = r
        data[solver] = {dim: [slot[k] for k in sorted(slot)] for dim, slot in sorted(merged.items())}

        total = sum(len(rows) for rows in data[solver].values())
        note = f' (+{added} new, {replaced} replaced)' if existing else ''
        print(f'  {solver:15s}: {len(data[solver]):2d} dims, {total:4d} runs{note}')

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
                             'e.g. --solvers dq_basic cayley')
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.out_dir, '_consolidated.npy')
    consolidate(args.out_dir, output_path, solvers=args.solvers)
