import os

# Must be set before jax is imported.
# JAX would otherwise reserve 75% of GPU memory up front, which also costs several GB
# of host RAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import time
from pathlib import Path

import dynamiqs as dq
import h5py
import numpy as np
import qutip as qt
import scqubits as scq

import floquet as ft
from floquet.compute_floquet import compute_floquet_grid

STATE_INDICES = [0, 1]

# Per-batch outputs of FloquetAnalysis.run() that are stored for the full grid.
# Values give the trailing shape: "states" -> (num_states,), "dim" -> (hilbert_dim,).
PER_POINT_OUTPUTS = {
    "bare_state_overlaps": "states",
    "intermediate_displaced_state_overlaps": "states",
    "batch_displaced_state_overlaps": "states",  # overlap with the per-batch fit
    "quasienergies": "dim",
    "avg_excitation": "dim",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IST Floquet simulation")
    # physics
    p.add_argument("--zeta", type=float, required=True)
    p.add_argument("--alpha_tilde", type=float, required=True)
    p.add_argument("--flux", type=float, required=True)
    p.add_argument("--omega_0", type=float, default=1.0)
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--hilbert_dim", type=int, required=True)
    # drive frequency grid, as ratios of omega_0
    p.add_argument("--freq_ratio_start", type=float, required=True)
    p.add_argument("--freq_ratio_end", type=float, required=True)
    p.add_argument("--num_freqs", type=int, required=True,
                   help="total number of drive frequencies")
    p.add_argument("--freqs_per_batch", type=int, required=True)
    # drive amplitude grid
    p.add_argument("--xi_sq_start", type=float, required=True)
    p.add_argument("--xi_sq_end", type=float, required=True)
    p.add_argument("--xi_sq_num", type=int, required=True)
    # analysis options
    p.add_argument("--fit_range_fraction", type=float, default=1.0)
    p.add_argument("--fit_cutoff", type=int, default=4)
    p.add_argument("--overlap_cutoff", type=float, default=0.8)
    p.add_argument("--max_steps", type=int, default=30_000)
    p.add_argument("--keep_floquet_modes", type=int, default=0,
                   help="1 to keep the Floquet modes in the final file")
    # files
    p.add_argument("--work_dir", type=Path, required=True,
                   help="node-local directory for the HDF5 files")
    p.add_argument("--output_name", type=str, required=True)
    return p.parse_args()


def frequency_grid(args: argparse.Namespace) -> tuple[np.ndarray, list[slice]]:
    """Uniform grid of num_freqs points over [start, end], cut into consecutive batches.

    Batches hold freqs_per_batch points each; the last batch also takes any remainder,
    so no batch is ever smaller than freqs_per_batch. Batches do not overlap. Only the
    last batch can differ in size, which costs at most one extra JAX compile.
    """
    if not 0 < args.freqs_per_batch <= args.num_freqs:
        raise ValueError("need 0 < freqs_per_batch <= num_freqs")
    ratios = np.linspace(args.freq_ratio_start, args.freq_ratio_end, args.num_freqs)
    num_batches = args.num_freqs // args.freqs_per_batch
    edges = [b * args.freqs_per_batch for b in range(num_batches)] + [args.num_freqs]
    return ratios, [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def ist_operators(args: argparse.Namespace) -> tuple[qt.Qobj, qt.Qobj, dict]:
    """H0 (diagonal, ground state at zero) and the charge operator."""
    omega_0 = args.omega_0
    alpha_tilde = args.alpha_tilde
    zeta = args.zeta
    eta = 8 * alpha_tilde/zeta 

    ist = scq.Fluxonium(
        EC=(omega_0 * zeta)/8,
        EJ=eta * omega_0/zeta,
        EL=(1 + eta) * omega_0/zeta,
        flux=args.flux,
        cutoff=args.cutoff,
        truncated_dim=args.hilbert_dim,
    )
    hilbert_space = scq.HilbertSpace([ist])
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0][: args.hilbert_dim]
    H0 = qt.Qobj(np.diag(evals - evals[0]))
    H1 = hilbert_space.op_in_dressed_eigenbasis(ist.n_operator)
    return H0, H1, ist.get_initdata()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    dq.set_device("gpu")
    dq.set_precision("double")

    ratios, batches = frequency_grid(args)
    num_batches = len(batches)
    omega_d = ratios * args.omega_0
    xi_sq = np.linspace(args.xi_sq_start, args.xi_sq_end, args.xi_sq_num)
    H0, H1, ist_initdata = ist_operators(args)
    amps = ft.XiSqToAmp(H0, H1, STATE_INDICES, omega_d).amplitudes_for_omega_d(xi_sq)

    n_freq, n_amp = len(omega_d), len(xi_sq)
    n_states, dim = len(STATE_INDICES), args.hilbert_dim
    batch = args.freqs_per_batch
    print(
        f"zeta={args.zeta} alpha_tilde={args.alpha_tilde}: {num_batches} batches "
        f"flux={args.flux} "
        f"(sizes {sorted({sl.stop - sl.start for sl in batches})}) over {n_freq} freqs, "
        f"{n_amp} amps, dim {dim} -> {n_freq * n_amp:,} points",
        flush=True,
    )

    options = ft.Options(
        fit_range_fraction=args.fit_range_fraction,
        fit_cutoff=args.fit_cutoff,
        overlap_cutoff=args.overlap_cutoff,
        max_steps=args.max_steps,
        save_floquet_modes=True,  # needed for the full-range refit
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.work_dir / args.output_name
    modes_path = args.work_dir / "floquet_modes_scratch.h5"

    trailing = {"states": (n_states,), "dim": (dim,)}
    with h5py.File(result_path, "w") as res, h5py.File(modes_path, "w") as scratch:
        
        res.attrs["zeta"] = args.zeta
        res.attrs["alpha_tilde"] = args.alpha_tilde
        res.attrs["flux"] = args.flux
        res.attrs["state_indices"] = STATE_INDICES
        res.attrs["num_batches"] = num_batches
        res.attrs["freqs_per_batch"] = batch
        res.attrs["args"] = json.dumps({k: str(v) for k, v in vars(args).items()})
        res.attrs["ist_initdata"] = json.dumps(ist_initdata, default=str)
        res["freq_ratios"] = ratios
        res["omega_d"] = omega_d
        res["xi_sq"] = xi_sq
        res["drive_amplitudes"] = amps  # (n_amp, n_freq)

        for name, kind in PER_POINT_OUTPUTS.items():
            shape = (n_freq, n_amp, *trailing[kind])
            res.create_dataset(name, shape=shape, dtype="f8", chunks=(batch, *shape[1:]))
        modes_shape = (n_freq, n_amp, n_states, dim)
        scratch.create_dataset("floquet_modes", shape=modes_shape, dtype="c16",
                               chunks=(batch, *modes_shape[1:]))

        batch_fits = []
        for b, sl in enumerate(batches):
            t_b = time.time()
            model = ft.Model(H0, H1, omega_d_values=omega_d[sl],
                             drive_amplitudes=amps[:, sl])
            data = ft.FloquetAnalysis(
                model, state_indices=STATE_INDICES, options=options
            ).run(filepath=None)

            for name in PER_POINT_OUTPUTS:
                src = "displaced_state_overlaps" if name.startswith("batch_") else name
                res[name][sl] = data[src]
            batch_fits.append(data["fit_data"])
            scratch["floquet_modes"][sl] = data["floquet_modes"]
            del data, model

            print(
                f"batch {b + 1}/{num_batches} done in {time.time() - t_b:.1f} s "
                f"(compiled grid versions: {compute_floquet_grid._cache_size()})",
                flush=True,
            )

        res["batch_fit_data"] = np.stack(batch_fits)  # (batch, state, dim, term)

        # refit the displaced states across the full frequency range
        t_fit = time.time()
        full_model = ft.Model(H0, H1, omega_d_values=omega_d, drive_amplitudes=amps)
        displaced_state = ft.DisplacedStateFit(
            hilbert_dim=dim, model=full_model,
            state_indices=STATE_INDICES, options=options,
        )
        floquet_modes = scratch["floquet_modes"][()]
        mask = res["intermediate_displaced_state_overlaps"][()]
        full_fit = displaced_state.displaced_states_fit(mask, floquet_modes)
        del mask
        res["fit_data"] = full_fit
        res["fit_ok"] = displaced_state.fit_ok
        res["displaced_state_overlaps"] = displaced_state.overlap_with_displaced_states(
            full_fit, floquet_modes
        )
        if args.keep_floquet_modes:
            res.create_dataset("floquet_modes", data=floquet_modes,
                               chunks=(batch, *modes_shape[1:]))
        del floquet_modes
        print(f"full-range refit done in {time.time() - t_fit:.1f} s", flush=True)

    modes_path.unlink()
    print(f"finished in {(time.time() - t_start) / 60:.1f} min -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
