import os
from threadpoolctl import threadpool_info

num_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
print(f'num_cpus = {num_cpus}')
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cpus)

import numpy as np
print(threadpool_info())
print()

###
import floquet as flq

base_path = 'out/'

###
sim_params = np.load(base_path + 'params_initial.npy', allow_pickle=True).item()

# state indices. Recall, they're saved as (transmon idx, resonator idx)
assert np.allclose(sim_params['assignment_idxs'][:, 0, 0], 0), \
    f"mismatched index |0_t, 0_r>. Actually = {sim_params['assignment_idxs'][:, 0, 0]}"
assert np.allclose(sim_params['assignment_idxs'][:, 1, 0], 1), \
    f"mismatched index |1_t, 0_r>. Actually = {sim_params['assignment_idxs'][:, 1, 0]}"
assert np.allclose(sim_params['assignment_idxs'][:, 0, 1], 2), \
    f"mismatched index |0_t, 1_r>. Actually = {sim_params['assignment_idxs'][:, 0, 1]}"
assert np.allclose(sim_params['assignment_idxs'][:, 1, 1], 4), \
    f"mismatched index |1_t, 1_r>. Actually = {sim_params['assignment_idxs'][:, 1, 1]}"

state_indices = [0, 1, 2, 4]

###
# Floquet modes, memmap'd
floquet_modes = np.load(base_path + 'floquet_modes_initial.npy', mmap_mode='r')
assert len(sim_params['omega_d_values']) == floquet_modes.shape[0], (
    f"Frequency count mismatch: params has {len(sim_params['omega_d_values'])}, "
    f"floquet_modes has {floquet_modes.shape[0]}."
)

###
# Only the transmon's mean excitation is needed to detect branch swaps.
avg_excitation = np.load(base_path + 'avg_excitation.npy', mmap_mode='r')
transmon_excitation = np.asarray(avg_excitation[:, :, 0, state_indices])

###
# At zero drive amplitude the Floquet mode IS the bare eigenstate.
assert np.all(sim_params['drive_amplitudes'][0] == 0), \
    "amp_idx 0 is not zero drive, so the undriven modes are not the bare states"
bare_states = np.asarray(floquet_modes[:, 0, :, :])

###
opts = flq.Options(fit_cutoff=8)

displaced_state = flq.DisplacedStateFit(
    omega_d_values=sim_params['omega_d_values'],
    drive_amplitudes=sim_params['drive_amplitudes'],
    bare_states=bare_states,
    state_indices=state_indices,
    options=opts,
)

coeffs, mask = displaced_state.displaced_states_fit(
    transmon_excitation, floquet_modes
)

###
np.save(f'{base_path}/coeffs.npy', coeffs)
np.save(f'{base_path}/mask.npy', mask)
np.save(f'{base_path}/exponent_pairs.npy', displaced_state.exponent_pairs)
np.save(f'{base_path}/poly_terms.npy', displaced_state.poly_terms)
