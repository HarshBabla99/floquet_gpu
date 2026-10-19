import os

# Must be set before jax is imported.
# JAX would otherwise reserve 75% of GPU memory up front, which also costs several GB
# of host RAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import resource
import time
from pathlib import Path

import jax
import numpy as np

import floquet as ft
from floquet.compute_floquet import compute_floquet_grid
from floquet.utils.file_io import open_h5_for_writing, write_datasets

from bif_ro_model import BifurcationROSystem
from params import params, tpi

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bifurcation readout Floquet simulation")

    # The physics parameters are in params.py
    # drive frequency batches
    p.add_argument("--freqs_per_batch", type=int, required=True,
                   help="number of drive frequencies per batch")

    # Floquet analysis options
    p.add_argument("--amp_chunk", type=int, default=4,
                   help="number of drive amplitudes solved at once on the GPU")
    p.add_argument("--fit_range_fraction", type=float, default=1.0,
                   help="fraction of the drive amplitudes to simultaneously fit the displaced state")
    p.add_argument("--fit_cutoff", type=int, default=8,
                   help="number of terms in the displaced state fit")
    p.add_argument("--overlap_cutoff", type=float, default=0.8,
                   help="cutoff for the overlap between the Floquet modes and the bare states, to specify the modes included within the fit")
    p.add_argument("--max_steps", type=int, default=30_000,
                   help="max steps for the sesolve")
    p.add_argument("--fit_freq_chunk", type=int, default=100,
                   help="number of drive frequencies per chunk for the online refit over the full range")
    p.add_argument("--keep_floquet_modes", type=int, default=0,
                   help="flag to save the Floquet modes in the final file")

    # files
    p.add_argument("--work_dir", type=Path, required=True,
                   help="node-local directory for the temporary and final files")
    p.add_argument("--output_name", type=str, required=True,
                   help="name of the final output file")
    return p.parse_args()


def memory_report() -> str:
    """Host RSS (now and peak) and GPU memory (in use and peak), for the log."""
    rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB on Linux
    report = f"host RSS {rss:.1f} GB (peak {rss_peak:.1f} GB)"
    stats = jax.devices()[0].memory_stats()
    if stats:
        report += (f", GPU {stats['bytes_in_use'] / 1e9:.1f} GB "
                   f"(peak {stats.get('peak_bytes_in_use', 0) / 1e9:.1f} GB)")
    return report


def frequency_batches(num_freqs: int, freqs_per_batch: int) -> list[slice]:
    """Split the frequency indices into batches. The last batch has any remaining
    unequally split points. This will cost at most one extra JAX compile.
    """
    # Validate that we have fewer freqs per batch than the total number of freqs
    if not 0 < freqs_per_batch <= num_freqs:
        raise ValueError("need 0 < freqs_per_batch <= num_freqs")

    num_batches = num_freqs // freqs_per_batch
    edges = [b * freqs_per_batch for b in range(num_batches)] + [num_freqs]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def main() -> None:
    args = parse_args()
    t_start = time.time()

    # Build the system: solve for the bare parameters at every resonator frequency
    system = BifurcationROSystem(**params)
    subsystem_dims = system.hilbert_dim[:-1]
    print(f"system set up in {time.time() - t_start:.1f} s | {memory_report()}", flush=True)

    # Compare the dressed parameters to the targets (first resonator frequency)
    omega_q, omega_r, alpha_q, kerr, chi = system.dressed_parameters[0] / tpi
    print(
        f"Dressed parameters at omega_r/2pi = {system.omega_r_values[0] / tpi:.3f} GHz \n"
        f"\t omega_q = {omega_q:.6f} GHz, alpha_q = {alpha_q:.6f} GHz \n"
        f"\t omega_r = {omega_r:.6f} GHz (target {system.omega_r_values[0] / tpi:.6f}) \n"
        f"\t Kerr    = {kerr:.6f} GHz (target {params['K'] / tpi:.6f}) \n"
        f"\t chi     = {chi:.6f} GHz (target {params['chi'] / tpi:.6f}) \n",
        flush=True,
    )

    # Track |0_t, 0_r>, |1_t, 0_r>, |0_t, 1_r>, |1_t, 1_r>. The model labels the dressed
    # states by product states, with index i * resonator_dim + j.
    state_indices = np.ravel_multi_index(([0, 1, 0, 1], [0, 0, 1, 1]), subsystem_dims).tolist()

    # split freq grid into batches
    batch_slices = frequency_batches(len(system.omega_d_values), args.freqs_per_batch)
    num_batches = len(batch_slices)

    # size of the axes
    n_freq, n_amp = system.drive_amplitudes.shape[1], system.drive_amplitudes.shape[0]
    n_states, hilbert_dim = len(state_indices), system.hilbert_dim[-1]
    n_excitations = len(subsystem_dims) + 1  # transmon, resonator, composite
    n_freq_per_batch = args.freqs_per_batch
    print(
        f"Bifurcation readout: hilbert_dim = {subsystem_dims} -> {hilbert_dim} \n"
        f"\t omega_r/2pi = {system.omega_r_values[0] / tpi:.3f}:{system.omega_r_values[-1] / tpi:.3f}:{n_freq} GHz \n"
        f"\t   {num_batches} batches "
        f"(sizes: {sorted({sl.stop - sl.start for sl in batch_slices})}) \n"
        f"\t photon numbers = {params['photon_nums'][0]:.3f}:{params['photon_nums'][-1]:.3f}:{n_amp} \n"
        f"\t state indices = {state_indices} \n"
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
        amp_chunk=args.amp_chunk,
        fit_freq_chunk=args.fit_freq_chunk,
    )

    # Creating working directory, path to store final results and to store the temp ones
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.work_dir / args.output_name
    scratch_path = args.work_dir / "scratch.h5"

    # H5 datasets
    def new_dataset(format: str, dtype: str = "f8") -> dict:
        match format:
            case "overlaps"    : shape = (n_freq, n_amp, n_states)
            case "scalars"     : shape = (n_freq, n_amp, hilbert_dim)
            case "excitations" : shape = (n_freq, n_amp, n_excitations, hilbert_dim)
            case "states"      : shape = (n_freq, n_amp, n_states, hilbert_dim)
        return {"shape": shape, "dtype": dtype, "chunks": (n_freq_per_batch, *shape[1:])}

    result_datasets = {
        "quasienergies": new_dataset("scalars"),
        "avg_excitation": new_dataset("excitations"),
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
        "state_indices": state_indices,
        "subsystem_dims": subsystem_dims,
        "num_batches": num_batches,
        "freqs_per_batch": n_freq_per_batch,
    } | {k: v for k, v in params.items() if k not in ("omega_r_values", "photon_nums")}
    drive_params = {
        "omega_r_values": system.omega_r_values,
        "omega_d_values": system.omega_d_values,
        "photon_nums": params["photon_nums"],
        "stark_shifts": system.stark_shifts,
        "drive_amplitudes": system.drive_amplitudes,
    }
    system_params = {
        "omega_r_optimized": system.omega_r_optimized,
        "kerr_optimized": system.kerr_optimized,
        "g_optimized": system.g_optimized,
        "undriven_evals": system.undriven_evals,
        "assignment_idxs": system.assignment_idxs,
        "dressed_parameters": system.dressed_parameters,
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
        write_datasets(result_file.create_group("system_params"), system_params)
        write_datasets(
            result_file.create_group("args"),
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )

        # iterate through batches
        for b, sl in enumerate(batch_slices):
            # run this batch, but don't save results
            t_b = time.time()
            model = system.model(sl)
            t_model = time.time() - t_b
            data = ft.FloquetAnalysis(
                model, state_indices=state_indices, options=options
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
                f"(model built in {t_model:.1f} s, "
                f"compiled grid versions: {compute_floquet_grid._cache_size()}) "
                f"| {memory_report()}",
                flush=True,
            )

        # refit the displaced states across the full frequency range
        # The fit only needs the drive grid and the bare states. These are the identity
        # in the dressed basis of every frequency, so a diagonal placeholder H0 suffices.
        t_fit = time.time()
        full_model = ft.Model(
            np.zeros((hilbert_dim, hilbert_dim)), np.zeros((hilbert_dim, hilbert_dim)),
            omega_d_values=system.omega_d_values, drive_amplitudes=system.drive_amplitudes,
        )
        displaced_state = ft.DisplacedStateFit(
            hilbert_dim=hilbert_dim, model=full_model,
            state_indices=state_indices, options=options,
        )
        # pass the h5py datasets, so the online fit only reads one chunk at a time
        floquet_modes = modes_file["floquet_modes"]
        ovlps_for_mask = scratch_file["displaced_state_overlaps"]
        full_fit = displaced_state.displaced_states_fit(ovlps_for_mask, floquet_modes)

        # compute overlaps with displaced states, one chunk of frequencies at a time
        displaced_state_overlaps = np.empty((n_freq, n_amp, n_states))
        for start in range(0, n_freq, args.fit_freq_chunk):
            stop = min(start + args.fit_freq_chunk, n_freq)
            displaced_state_overlaps[start:stop] = (
                displaced_state.overlap_with_displaced_states(
                    full_fit, floquet_modes, omega_d_idxs=(start, stop)
                )
            )

        write_datasets(result_file, {
            "fit_data": full_fit,
            "fit_ok": displaced_state.fit_ok,
            "displaced_state_overlaps": displaced_state_overlaps,
        })

        print(f"\nfull-range refit done in {time.time() - t_fit:.1f} s | {memory_report()}",
              flush=True)

    scratch_path.unlink()
    print(f"finished in {(time.time() - t_start) / 60:.1f} min -> {result_path} "
          f"| {memory_report()}", flush=True)


if __name__ == "__main__":
    main()
