from __future__ import annotations

import time

import numpy as np

from scipy.optimize import linear_sum_assignment

from .displaced_state import DisplacedState, DisplacedStateFit
from .model import Model
from .options import Options
from .utils.file_io import Serializable
from .compute_floquet import compute_floquet_grid


class FloquetAnalysis(Serializable):
    """Perform a floquet analysis to identify nonlinear resonances.

    In most workflows, one needs only to call the run() method which performs
    both the displaced state fit and the Blais branch analysis. For an example
    workflow, see the [transmon](../examples/transmon) tutorial.

    Parameters:
        model: Class specifying the model, including the Hamiltonian, drive amplitudes,
            frequencies
        state_indices: State indices of interest. Defaults to [0, 1], indicating the two
            lowest-energy states.
        options: Options for the Floquet analysis.
            ??? info "Detailed `opt_options` API"
                - fit_range_fraction (`float`, default: 1.0): Fraction of the amplitude
                    range to sweep over before changing the definition of the bare state
                    to that of the itted state from the previous range. For instance if
                    fit_range_fraction=0.4, then the amplitude range is split up into
                    three chunks: the first 40% of the amplitude linspace, then from
                    40% -> 80%, then from 80% to the full range. For the first fraction
                    of amplitudes, they are compared to the bare eigenstates for
                    identification. For the second range, they are compared to the
                    fitted state from the first range. And so on. Defaults to 1.0,
                    indicating that no iteration is performed.
                - floquet_sampling_time_fraction (`float`, default: 0.0): What point of
                    the drive period we want to sample the Floquet modes. Defaults to
                    0.0, indicating the floquet modes at t=0*T where T is the drive
                    period.
                - fit_cutoff (`int`, default: 4): Cutoff for the fit polynomial of the
                    displaced state.
                - overlap_cutoff (`float`, default: 0.8): Cutoff for fitting overlaps.
                    Floquet modes with overlap with the "bare" state below this cutoff
                    are not included in the fit (as they may be experiencing a
                    resonance).
                - nsteps (`int`, default: 30_000): QuTiP integration parameter, number
                    of steps the solver can take.
                - num_cpus (`int`, default: 1): Number of cpus to use in parallel
                    computation of Floquet modes over the different values of
                    omega_d, amp.
                - save_floquet_modes (`bool`, default: False): Indicating whether to
                    save the extracted Floquet modes themselves. Such data is often
                    unnecessary and requires a fair amount of storage, so the default is
                    False.
        init_data_to_save: Initial parameter metadata to save to file. Defaults to None.
    """

    def __init__(
        self,
        model: Model,
        state_indices: list | None = None,
        options: Options = Options(),  # noqa B008
        init_data_to_save: dict | None = None,
    ):
        if state_indices is None:
            state_indices = [0, 1]
        self.model = model
        self.state_indices = state_indices
        self.options = options
        self.init_data_to_save = init_data_to_save
        self.hilbert_dim = model.hilbert_dim

    def __str__(self) -> str:
        return "Running floquet simulation with parameters: \n" + super().__str__()

    @staticmethod
    def _linear_sum_assignment(score: np.ndarray) -> np.ndarray:
        """One-to-one row->column assignment maximising total score.
        Prefers numpy's argmax if possible, otherwise falls back to scipy's 
        linear_sum_assignment.
        
        Parameters:
            score: (..., R, C) with R <= C.
            
        Returns:
            column indices of shape (..., R).
        """
        idx = np.argmax(score, axis=-1)
        s = np.sort(idx, axis=-1)
        clash = np.any(s[..., 1:] == s[..., :-1], axis=-1)
        for b in zip(*np.nonzero(clash)):
            idx[b] = linear_sum_assignment(-score[b])[1]
        return idx


    def identify_floquet_modes(
        self,
        floquet_modes: np.ndarray,
        displaced_state: DisplacedState,
        previous_coefficients: np.ndarray,
        amp_idx_0: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return floquet modes with largest overlap with ideal displaced state.
        Batched over all the drive frequencies. 

        Also return that overlap value.

        Parameters:
            floquet_modes: Shape: `(num_omega_ds, num_amps, hilbert_dim, hilbert_dim)`,
                where the first index labels the mode and the second the basis vector
                components.
            displaced_state: Instance of DisplacedState.
            previous_coefficients: Coefficients from the previous amplitude range that
                will be used when calculating overlap of the floquet modes against
                the 'bare' states specified by previous_coefficients.
            amp_idx_0: Index specifying the lower bound of the amplitude range.
                0 by default, i.e. selects the undriven states.

        Returns:
            bare_state_overlaps: Shape `(num_omega_ds, num_amps, num_states)`
            floquet_modes: Shape `(num_omega_ds, num_amps, num_states, hilbert_dim)`
                where the modes are ordered by the largest overlap with the ideal displaced state.
        """

        # Compute the overlap of the floquet modes with the displaced states specified
        # Result = (num_omega_ds, self.state_indices, self.hilbert_dim)
        displaced_states = displaced_state.displaced_state(
            previous_coefficients,
            amp_idxs=[amp_idx_0, amp_idx_0 + 1],
        )[:, 0, ...]
        overlaps = np.einsum("wsh,wath->wast", np.conj(displaced_states), floquet_modes)

        # Enforce one-to-one assignment between tracked states and Floquet modes.
        f_idxs = self._linear_sum_assignment(np.abs(overlaps))
        bare_state_overlaps = np.take_along_axis(overlaps, f_idxs[..., None], axis=-1)[..., 0]
        picked = np.take_along_axis(floquet_modes, f_idxs[..., None], axis=2)

        # Fix phase uncertainty by ensuring that the bare state overlap is real and positive.
        # Return bare state overlaps and the corresponding floquet modes, with the phase adjusted.
        return (
            np.abs(bare_state_overlaps),
            picked / np.sign(bare_state_overlaps)[..., None]
        )

    def branch_analysis(
        self, 
        floquet_modes: np.ndarray, 
        quasienergies: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform Blais branch analysis.

        Gorgeous in its simplicity. Simply calculate overlaps of new floquet modes with
        those from the previous amplitude step, and order the modes accordingly.
        """
        ordered_modes = np.empty_like(floquet_modes)
        ordered_quasienergies = np.empty_like(quasienergies)
        num_amps = floquet_modes.shape[1]

        # since we're sorting all modes, begin with the bare states
        prev_modes = np.broadcast_to(
            self.model.bare_state_array(), 
            (floquet_modes.shape[0], self.hilbert_dim, self.hilbert_dim)
        )
        for amp_idx in range(num_amps):
            ovlps = np.abs(np.einsum("wih,wkh->wik", prev_modes.conj(), floquet_modes[:, amp_idx]))
            perm = self._linear_sum_assignment(ovlps)
            prev_modes = np.take_along_axis(floquet_modes[:, amp_idx], perm[..., None], axis=1)
            ordered_modes[:, amp_idx] = prev_modes
            ordered_quasienergies[:, amp_idx] = np.take_along_axis(
                quasienergies[:, amp_idx], perm, axis=1
            )

        return ordered_modes, ordered_quasienergies

    def _calculate_mean_excitation(self, f_modes_ordered: np.ndarray) -> np.ndarray:
        """Mean excitation number of ordered floquet modes.

        Based on Blais arXiv:2402.06615, specifically Eq. (12) but going without the
        integral over floquet modes in one period.
        """
        bare = self.model.bare_state_array()
        overlaps_sq = np.abs(np.einsum("ih,...kh->...ik", bare, f_modes_ordered)) ** 2

        # sum over bare excitations weighted by excitation number
        return np.real(
            np.einsum("...ik,i->...k", overlaps_sq, np.arange(self.hilbert_dim))
        )

    def run(self, filepath: str | None = None) -> dict:
        """Perform floquet analysis over range of amplitudes and drive frequencies.

        This function largely performs two calculations. The first is the Xiao analysis
        introduced in https://arxiv.org/abs/2304.13656, fitting the extracted Floquet
        modes to the "ideal" displaced state which does not include resonances by design
        (because we fit to a low order polynomial and ignore any floquet modes with
        overlap with the bare state below a given threshold). This analysis produces the
        "scar" plots. The second is the Blais branch analysis, which tracks the Floquet
        modes by stepping in drive amplitude for a given drive frequency. For this
        reason the code is structured to parallelize over drive frequency, but scans in
        a loop over drive amplitude. This way the two calculations can be performed
        simultaneously.

        A nice bonus is that both of the above mentioned calculations determine
        essentially independently whether a resonance occurs. In the first, it is
        deviation of the Floquet mode from the fitted displaced state. In the second,
        it is branch swapping that indicates a resonance, independent of any fit. Thus
        the two simulations can be used for cross validation of one another.

        We perform these simulations iteratively over the drive amplitudes as specified
        by fit_range_fraction. This is to allow for simulations stretching to large
        drive amplitudes, where the overlap with the bare eigenstate would fall below
        the threshold (due to ac Stark shift) even in the absence of any resonances.
        We thus use the fit from the previous range of drive amplitudes as our new bare
        state.
        """
        # print(self)
        start_time = time.time()

        # shapes for the arrays that will contain our data
        num_omega_d = len(self.model.omega_d_values)
        num_amps = len(self.model.drive_amplitudes)
        num_states = len(self.state_indices)

        # Solve the Floquet problem for all the drive parameters
        f_modes, f_energies = compute_floquet_grid(self.model, options=self.options)
        f_modes, f_energies = np.asarray(f_modes), np.asarray(f_energies)

        # Blais branch analysis. Independent of the fit ranges, so done in one pass.
        blais_ordered_modes, quasienergies = self.branch_analysis(f_modes, f_energies)
        avg_excitation = self._calculate_mean_excitation(blais_ordered_modes)

        # Displaced state analysis. Iterated over amplitude ranges.
        displaced_state = DisplacedStateFit(
            hilbert_dim=self.hilbert_dim,
            model=self.model,
            state_indices=self.state_indices,
            options=self.options,
        )
        previous_coefficients = np.zeros(
            (num_states, self.hilbert_dim, displaced_state.exponent_pairs.shape[-1]),
            dtype=complex,
        )
        bare_state_overlaps = np.zeros((num_omega_d, num_amps, num_states))
        intermediate_displaced_state_overlaps = np.zeros_like(bare_state_overlaps)
        floquet_modes = np.zeros(
            (num_omega_d, num_amps, num_states, self.hilbert_dim), dtype=complex
        )

        # Now iterate through the amplitude ranges
        num_fit_ranges = int(np.ceil(1 / self.options.fit_range_fraction))
        num_amp_pts_per_range = int(
            np.floor(len(self.model.drive_amplitudes) / num_fit_ranges)
        )
        for amp_range_idx in range(num_fit_ranges):
            print(f"calculating for amp_range_idx={amp_range_idx}")
            # edge case if range doesn't fit in neatly
            if amp_range_idx == num_fit_ranges - 1:
                amp_range_idx_final = len(self.model.drive_amplitudes)
            else:
                amp_range_idx_final = (amp_range_idx + 1) * num_amp_pts_per_range
            amp_idxs = [amp_range_idx * num_amp_pts_per_range, amp_range_idx_final]
            amp_slice = slice(amp_idxs[0], amp_idxs[1])

            # Identify the floquet modes wrt the the bare-like state computed from the prev coeffs
            bare_state_overlaps[:, amp_slice], floquet_modes[:, amp_slice] = (
                self.identify_floquet_modes(
                    f_modes[:, amp_slice], displaced_state, previous_coefficients, amp_idxs[0]
                )
            )

            # ovlp_with_bare_states is used as a mask for the fit
            ovlp_with_bare_states = displaced_state.overlap_with_bare_states(
                previous_coefficients, floquet_modes[:, amp_slice], amp_idx_0=amp_idxs[0]
            )
            # Compute the fitted 'ideal' displaced state, excluding those
            # floquet modes experiencing resonances.
            new_coefficients = displaced_state.displaced_states_fit(
                ovlp_with_bare_states, floquet_modes[:, amp_slice], amp_idxs=amp_idxs
            )
            # Compute overlap of floquet modes with ideal displaced state using this
            # new fit. We use this data as the mask for when we compute the coefficients
            # over the whole range.
            intermediate_displaced_state_overlaps[:, amp_slice] = (
                displaced_state.overlap_with_displaced_states(
                    new_coefficients, floquet_modes, amp_idxs=amp_idxs
                )
            )

            # Keep the last good coefficients for any state whose fit failed.
            fit_ok = displaced_state.fit_ok
            previous_coefficients[fit_ok] = new_coefficients[fit_ok]

        # The previously extracted coefficients were valid for the amplitude ranges
        # we asked for the fit over. Now armed with with correctly identified floquet
        # modes, we recompute these coefficients over the whole sea of floquet mode data
        # to get a plot that is free from numerical artifacts associated with
        # the fits being slightly different at the boundary of ranges. We utilize the
        # previously computed overlaps of the floquet modes with the displaced states
        # (stored in intermediate_displaced_state_overlaps) to obtain the mask with
        # which we exclude some data from the fit (because we suspect they've hit
        # resonances).
        full_displaced_fit = displaced_state.displaced_states_fit(
            intermediate_displaced_state_overlaps, floquet_modes
        )
        true_overlaps = displaced_state.overlap_with_displaced_states(
            full_displaced_fit, floquet_modes
        )
        data_dict = {
            "bare_state_overlaps": bare_state_overlaps,
            "fit_data": full_displaced_fit,
            "displaced_state_overlaps": true_overlaps,
            "intermediate_displaced_state_overlaps": intermediate_displaced_state_overlaps,  # noqa E501
            "quasienergies": quasienergies,
            "avg_excitation": avg_excitation,
        }
        if self.options.save_floquet_modes:
            data_dict["floquet_modes"] = floquet_modes
        print(f"finished in {(time.time() - start_time) / 60} minutes")
        if filepath is not None:
            self.write_to_file(filepath, data_dict)
        return data_dict
