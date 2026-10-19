import argparse
import os

import numpy as np

from params import params as sim_params, tpi

# Arrays produced by each mode.
ARRAYS = {
    'initial': ('avg_excitation', 'quasienergies', 'floquet_modes_initial',
                'undriven_evecs'),
    'final': ('displaced_state_overlap', 'floquet_modes_final'),
}

# Entries of params_<mode>.npy. 
PER_FREQ = ('omega_r_values', 'omega_d_values', 'omega_r_optimized',
            'kerr_optimized', 'g_optimized', 'undriven_evals', 'assignment_idxs',
            'dressed_parameters')
CONSTANT = ('stark_shifts', 'hilbert_dim', 'state_indices', 'photon_nums',
            'EJ', 'EC', 'ng', 'ncut', 'transmon_dim', 'K', 'delta_const',
            'resonator_dim', 'chi')
# drive_amplitudes is (num_amps, num_omega_d), so frequency is axis 1


####################################################################################################
def main(mode, parent_path, lo, hi, purge):
    freqs = sim_params['omega_r_values'] / tpi
    if hi is None:
        hi = len(freqs) - 1
    freq_idxs = range(lo, hi + 1)
    freq_dirs = [f'{parent_path}/freq_{freqs[i]:.3f}' for i in freq_idxs]

    missing = [d for d in freq_dirs if not os.path.isdir(d)]
    assert not missing, f'{len(missing)} frequency directories are missing, e.g. {missing[0]}'

    names = ARRAYS[mode]

    # merged = dict of merged arrays, where each array has shape (num_freqs, *per_freq_shape). 
    # the array is preallocated on disk as a memmap, so it can be larger than RAM.
    merged = {}
    for name in names:
        first = np.load(f'{freq_dirs[0]}/{name}.npy', mmap_mode='r')
        filename = f'{parent_path}/{name}.npy'
        merged[name] = np.lib.format.open_memmap(
            filename, mode='w+', dtype=first.dtype,
            shape=(len(freq_dirs), *first.shape),
        )
        print(f'allocated {filename} {merged[name].shape} {merged[name].dtype}', flush=True)

    # params = dict of params, where the params vary with freq, and need to be merged.
    # constants = dict of params, where the params are constant with freq, and need to be copied.
    params = {key: [] for key in (*PER_FREQ, 'drive_amplitudes')}
    constants = {}

    # Iterate through freq directories and merge the data
    # If a per-freq run dies then <name>.npy.part is left, instead of <name>.npy.
    # This shows up here as a missing file instead of zeros.
    for out_idx, freq_dir in enumerate(freq_dirs):
        print(freq_dir, flush=True)

        # merge the results from this freq
        for name in names:
            source = np.load(f'{freq_dir}/{name}.npy', mmap_mode='r')
            assert source.shape == merged[name].shape[1:], \
                f'{freq_dir}/{name}.npy has shape {source.shape}, expected {merged[name].shape[1:]}'
            merged[name][out_idx] = source

        # load the params
        _params = np.load(f'{freq_dir}/params_{mode}.npy', allow_pickle=True).item()

        # verify that there aren't any unfamiliar params
        unhandled = set(_params) - set(PER_FREQ) - set(CONSTANT) - {'drive_amplitudes'}
        assert not unhandled, (
            f'{freq_dir}/params_{mode}.npy holds {sorted(unhandled)}, which merge.py '
            f'does not know how to combine -- add each to PER_FREQ or CONSTANT'
        )

        # merge the params that vary with freq
        for key in params:
            params[key].append(_params[key])

        # copy the params that are constant with freq (only need to do this once)
        if out_idx == 0:
            constants = {key: _params[key] for key in CONSTANT}

        if purge:
            # Only safe because the data above is already on disk in the memmap.
            for name in names:
                os.remove(f'{freq_dir}/{name}.npy')
            os.remove(f'{freq_dir}/params_{mode}.npy')
            if not os.listdir(freq_dir):
                os.rmdir(freq_dir)

    for name in names:
        merged[name].flush()

    # Concat params. drive amplitudes concat along axis=1, all others along axis=0.
    drive_amplitudes = np.concatenate(params.pop('drive_amplitudes'), axis=1)
    params = {key: np.concatenate(vals, axis=0) for key, vals in params.items()}
    params['drive_amplitudes'] = drive_amplitudes
    params |= constants

    # verify that the merged values are the same as the original ones in params.py
    assert np.allclose(params['omega_r_values'], sim_params['omega_r_values'][lo:hi + 1]), \
        'merged omega_r_values do not match params.py -- wrong --lo/--hi, or stale data'

    # save the params (the merged arrays are already saved as memmaps)
    filename = f'{parent_path}/params_{mode}.npy'
    np.save(filename, params)
    print(f'saved {filename}')
    print('complete')


####################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bifurcation Readout")
    parser.add_argument("--mode", default='initial', type=str, help="Mode (initial or final)")
    parser.add_argument("--parent_path", default='out', type=str, help="Directory holding freq_*/")
    parser.add_argument("--lo", default=0, type=int, help="First freq index in params.py to merge")
    parser.add_argument("--hi", default=None, type=int, help="Last freq index (inclusive)")
    parser.add_argument("--purge", action='store_true',
                        help="Delete each frequency's files once merged")
    parser_args = vars(parser.parse_args())
    main(**parser_args)
