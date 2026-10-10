import os

# Must be set before jax is imported.
# JAX would otherwise reserve 75% of GPU memory up front, which also costs several GB
# of host RAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import time
from pathlib import Path

import numpy as np
import qutip as qt
import scqubits as scq

import floquet as ft
from floquet.compute_floquet import compute_floquet_grid
from floquet.utils.file_io import open_h5_for_writing, write_datasets

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IST Floquet simulation")
    
    # physics
    p.add_argument("--zeta", type=float, required=True,
                   help="dimensionless impedance of the IST (=sqrt{8 E_C/(E_L - E_J)} at half flux)")
    p.add_argument("--alpha_tilde", type=float, required=True,
                    help="anharmonicity of the IST normalized by the qubit frequency (=sqrt{E_C/8}* E_J * (E_L - E_J)^{-3/2} )")
    p.add_argument("--flux", type=float, required=True,
                   help="flux bias")
    p.add_argument("--omega_0", type=float, default=1.0,
                   help="qubit frequency of the IST (=sqrt{8 (E_L - E_J) E_C} at half flux)")
    p.add_argument("--cutoff", type=int, required=True,
                   help="number of bosonic states to diagonalize the static Hamiltonian")
    p.add_argument("--hilbert_dim", type=int, required=True,
                   help="number of IST eigenstates to keep after diagonalization")
    p.add_argument("--state_indices", type=lambda s: [int(x) for x in s.split(",")], 
                   required=False, default=[0,1],
                   help="indices of the Floquet modes to track")
    
    # drive frequency grid, as ratios of omega_0
    p.add_argument("--freq_ratio_start", type=float, required=True,
                   help="start of the omega_d/omega_0 interval")
    p.add_argument("--freq_ratio_end", type=float, required=True,
                   help="end of the omega_d/omega_0 interval")
    p.add_argument("--freq_ratio_num", type=int, required=True,
                   help="total number of omega_d/omega_0 points")
    p.add_argument("--freq_ratio_num_per_batch", type=int, required=True,
                   help="number of omega_d/omega_0 points per batch")
    
    # drive amplitude grid
    p.add_argument("--xi_sq_start", type=float, required=True,
                   help="start of the xi^2 interval")
    p.add_argument("--xi_sq_end", type=float, required=True,
                   help="end of the xi^2 interval")
    p.add_argument("--xi_sq_num", type=int, required=True,
                   help="number of xi^2 points")
    
    # Floquet analysis options
    p.add_argument("--fit_range_fraction", type=float, default=1.0,
                   help="fraction of the drive amplitudes to simultaneously fit the displaced state")
    p.add_argument("--fit_cutoff", type=int, default=4,
                   help="number of terms in the displaced state fit")
    p.add_argument("--overlap_cutoff", type=float, default=0.8,
                   help="cutoff for the overlap between the Floquet modes and the bare states, to specify the modes included within the fit")
    p.add_argument("--max_steps", type=int, default=30_000,
                   help="max steps for the sesolve")
    p.add_argument("--keep_floquet_modes", type=int, default=0,
                   help="flag to save the Floquet modes in the final file")
    
    # files
    p.add_argument("--work_dir", type=Path, required=True,
                   help="node-local directory for the temporary and final files")
    p.add_argument("--output_name", type=str, required=True,
                   help="name of the final output file")
    return p.parse_args()


def frequency_grid(args: argparse.Namespace) -> tuple[np.ndarray, list[slice]]:
    """Split frequency grid into batches. The last grid has any remaining unequally split 
    points. This will cost at most one extra JAX compile.
    """
    
    # Validate that we have fewer freqs per batch than the total number of freqs
    if not 0 < args.freq_ratio_num_per_batch <= args.freq_ratio_num:
        raise ValueError("need 0 < freqs_per_batch <= num_freqs")

    freq_ratios = np.linspace(args.freq_ratio_start, args.freq_ratio_end, args.freq_ratio_num)
    num_batches = args.freq_ratio_num // args.freq_ratio_num_per_batch
    edges = [b * args.freq_ratio_num_per_batch for b in range(num_batches)] + [args.freq_ratio_num]

    # return the list of frequencies and idx slices for the batches
    return freq_ratios, [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def ist_operators(args: argparse.Namespace) -> tuple[qt.Qobj, qt.Qobj, dict]:
    """Construct the drift and drive operator for the IST. 
    
    Returns: 
        H0: diagonal, ground state at zero.
        H1: charge operator
        init_data: the tmon params, as stored by scqubits
    """
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

    # split freq grid into batches
    freq_ratios, batch_slices = frequency_grid(args)
    num_batches = len(batch_slices)
    omega_d = freq_ratios * args.omega_0

    # get the xi^2 values, transmon ops, and drive frequencies
    xi_sq = np.linspace(args.xi_sq_start, args.xi_sq_end, args.xi_sq_num)
    H0, H1, ist_initdata = ist_operators(args)
    amps = ft.XiSqToAmp(H0, H1, args.state_indices, omega_d).amplitudes_for_omega_d(xi_sq)

    # size of the axes
    n_freq, n_amp = len(omega_d), len(xi_sq)
    n_states, hilbert_dim = len(args.state_indices), args.hilbert_dim
    n_freq_per_batch = args.freq_ratio_num_per_batch
    print(
        f"IST: "
        f"zeta = {args.zeta} alpha_tilde={args.alpha_tilde}, flux={args.flux} \n"
        f"\t freq_ratios = {freq_ratios[0]:.3f}:{freq_ratios[-1]:.3f}:{n_freq} \n"
        f"\t   {num_batches} batches "
        f"(sizes: {sorted({sl.stop - sl.start for sl in batch_slices})}) \n"
        f"\t xi^2 values = {xi_sq[0]:.3f}:{xi_sq[-1]:.3f}:{n_amp} \n"
        f"\t overall {n_freq * n_amp:,} points\n",
        flush=True,
    )

    # Floquet options
    options = ft.Options(
        fit_range_fraction=args.fit_range_fraction,
        fit_cutoff=args.fit_cutoff,
        overlap_cutoff=args.overlap_cutoff,
        max_steps=args.max_steps,
        save_floquet_modes=True,
    )

    # Creating working directory, path to store final results and to store the temp ones
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.work_dir / args.output_name
    scratch_path = args.work_dir / "scratch.h5"

    # H5 datasets
    def new_dataset(format: str, dtype: str = "f8") -> dict:
        match format:
            case "overlaps" : shape = (n_freq, n_amp, n_states)
            case "scalars"  : shape = (n_freq, n_amp, hilbert_dim)
            case "states"   : shape = (n_freq, n_amp, n_states, hilbert_dim)
        return {"shape": shape, "dtype": dtype, "chunks": (n_freq_per_batch, *shape[1:])}

    result_datasets = {
        "quasienergies": new_dataset("scalars"),
        "avg_excitation": new_dataset("scalars"),
    }
    scratch_datasets = {
        "displaced_state_overlaps": new_dataset("overlaps")
    }

    # we need the Floquet modes in scratch, 
    # but write them to the final dataset directly, if specified
    modes_datasets = result_datasets if args.keep_floquet_modes else scratch_datasets
    modes_datasets["floquet_modes"] = new_dataset("states", "c16")

    # save the parameters 
    static_params = {
        "zeta": args.zeta,
        "alpha_tilde": args.alpha_tilde,
        "flux": args.flux,
        "state_indices": args.state_indices,
        "num_batches": num_batches,
        "freqs_per_batch": n_freq_per_batch
    } | ist_initdata
    drive_params = {
        "freq_ratios": freq_ratios,
        "omega_d": omega_d,
        "xi_sq": xi_sq,
        "drive_amplitudes": amps,
    }

    # Open two files, one for the results and another as scratch
    with (
        open_h5_for_writing(result_path, empty_datasets=result_datasets) as result_file,
        open_h5_for_writing(scratch_path, empty_datasets=scratch_datasets) as scratch_file,
    ):
        # get the file to save the modes
        modes_file = result_file if args.keep_floquet_modes else scratch_file

        # save the params to the final results file
        write_datasets(result_file.create_group("drive_params"), drive_params)
        write_datasets(result_file.create_group("static_params"), static_params)
        write_datasets(
            result_file.create_group("args"),
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )

        # iterate through batches
        for b, sl in enumerate(batch_slices):
            # run this batch, but don't save results
            t_b = time.time()
            model = ft.Model(H0, H1, omega_d_values=omega_d[sl],
                             drive_amplitudes=amps[:, sl])
            data = ft.FloquetAnalysis(
                model, state_indices=args.state_indices, options=options
            ).run(filepath=None)

            # qenergies and avg_excitations go directly to the final result
            result_file["quasienergies"][sl] = data["quasienergies"]
            result_file["avg_excitation"][sl] = data["avg_excitation"]

            # displaced_state_overlaps are stored in scratch, for the final mask 
            scratch_file["displaced_state_overlaps"][sl] = data["displaced_state_overlaps"]

            # Floquet modes are saved to the relevant dataset
            modes_file["floquet_modes"][sl] = data["floquet_modes"]

            del data, model

            print(
                f"batch {b + 1}/{num_batches} done in {time.time() - t_b:.1f} s "
                f"(compiled grid versions: {compute_floquet_grid._cache_size()})",
                flush=True,
            )

        # refit the displaced states across the full frequency range
        t_fit = time.time()
        full_model = ft.Model(H0, H1, omega_d_values=omega_d, drive_amplitudes=amps)
        displaced_state = ft.DisplacedStateFit(
            hilbert_dim=hilbert_dim, model=full_model,
            state_indices=args.state_indices, options=options,
        )
        floquet_modes = modes_file["floquet_modes"][()]
        ovlps_for_mask = scratch_file["displaced_state_overlaps"][()]
        full_fit = displaced_state.displaced_states_fit(ovlps_for_mask, floquet_modes)
        del ovlps_for_mask
        
        # compute overlaps with displaced states and save
        write_datasets(result_file, {
            "fit_data": full_fit,
            "fit_ok": displaced_state.fit_ok,
            "displaced_state_overlaps": displaced_state.overlap_with_displaced_states(
                full_fit, floquet_modes
            ),
        })
        del floquet_modes

        print(f"\nfull-range refit done in {time.time() - t_fit:.1f} s", flush=True)

    scratch_path.unlink()
    print(f"finished in {(time.time() - t_start) / 60:.1f} min -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
