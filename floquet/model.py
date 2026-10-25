from __future__ import annotations

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
from scipy.optimize import linear_sum_assignment


class Model(eqx.Module):
    """Specify the model, including the Hamiltonian, drive strengths and frequencies.

    Can be subclassed to e.g. override the hamiltonian() method for a different (but
    still periodic!) Hamiltonian.

    Parameters:
        H0: Drift Hamiltonian, provided in units such that H0 can be passed directly
            to qutip. If H0 is not diagonal, it is diagonalized and the simulation is
            performed in its eigenbasis (see diagonalize). H0 can also depend on the
            drive frequency, in which case it has shape (num_omega_ds, n, n).
        H1: Drive operator, which should be unitless (for instance the charge-number
            operator n of the transmon). It will be multiplied by a drive amplitude
            that we scan over from drive_parameters.drive_amplitudes. Same basis as
            H0, with shape (n, n) or (num_omega_ds, n, n).
        omega_d_values: drive frequencies to scan over
        drive_amplitudes: amp values to scan over. Can be one dimensional in which case
            these amplitudes are used for all omega_d, or it can be two dimensional
            in which case the first dimension are the amplitudes to scan over
            and the second are the amplitudes for respective drive frequencies

    """

    # array fields are pytree leaves (traced under jax.jit); hilbert_dim is static
    omega_d_values: np.ndarray
    drive_amplitudes: np.ndarray
    H0_evals: jax.Array
    H1_tilde: jax.Array
    H0_evecs: jax.Array | None
    hilbert_dim: int = eqx.field(static=True)

    def __init__(
        self,
        H0: dq.QArray | qt.Qobj | np.ndarray | jnp.Array | list,
        H1: dq.QArray | qt.Qobj | np.ndarray | jnp.Array | list,
        omega_d_values: np.ndarray | list,
        drive_amplitudes: np.ndarray | list,
    ):
        if not isinstance(H0, dq.QArray):
            H0 = dq.asqarray(H0)
        if not isinstance(H1, dq.QArray):
            H1 = dq.asqarray(H1)
        if isinstance(omega_d_values, list):
            omega_d_values = np.array(omega_d_values)
        if isinstance(drive_amplitudes, list):
            drive_amplitudes = np.array(drive_amplitudes)
        if len(drive_amplitudes.shape) == 1:
            drive_amplitudes = np.tile(drive_amplitudes, (len(omega_d_values), 1)).T
        assert len(drive_amplitudes.shape) == 2
        assert drive_amplitudes.shape[1] == len(omega_d_values)

        self.omega_d_values = omega_d_values
        self.drive_amplitudes = drive_amplitudes

        # The simulation is performed in the eigenbasis of the drift Hamiltonian
        H0, H1 = np.asarray(H0.to_jax()), np.asarray(H1.to_jax())
        H0_diag = np.einsum("...ii->...i", H0)
        if np.array_equal(H0, H0_diag[..., None] * np.eye(H0.shape[-1])):
            # H0 is already diagonal, so the bare states are the basis states
            self.H0_evals = jnp.asarray(H0_diag)
            self.H1_tilde = jnp.asarray(H1)
            self.H0_evecs = None
        else:
            evals, evecs = self.diagonalize(H0)
            self.H0_evals = jnp.asarray(evals)
            self.H1_tilde = jnp.asarray(np.conj(np.swapaxes(evecs, -1, -2)) @ H1 @ evecs)
            self.H0_evecs = jnp.asarray(evecs)
        self.hilbert_dim = H0.shape[-1]

    @staticmethod
    def diagonalize(H0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition of H0 (shape (..., n, n)), labelled by the basis states.

        The eigenvectors are ordered one-to-one by the basis state they overlap most,
        and their phase is fixed such that this overlap is real and positive. So
        evecs[..., :, k] is the dressed version of basis state k, and the bare states
        are the identity in this eigenbasis. Unlike ordering by energy, this labelling
        does not jump as levels cross when H0 changes with the drive frequency.
        """
        evals, evecs = np.linalg.eigh(H0)
        for idx in np.ndindex(H0.shape[:-2]):
            # perm[k] is the eigenvector assigned to basis state k
            perm = linear_sum_assignment(-np.abs(evecs[idx]))[1]
            evals[idx], evecs[idx] = evals[idx][perm], evecs[idx][:, perm]
        overlaps = np.einsum("...kk->...k", evecs)
        evecs = evecs * np.conj(overlaps / np.abs(overlaps))[..., None, :]
        return evals, evecs

    def H0_evals_at(self, freq_idx: int = 0) -> jax.Array:
        """Eigenvalues of H0 at drive frequency index freq_idx. Shape: (hilbert_dim,)."""
        return self.H0_evals if self.H0_evals.ndim == 1 else self.H0_evals[freq_idx]

    def H1_tilde_at(self, freq_idx: int = 0) -> jax.Array:
        """Drive operator in the H0 eigenbasis at drive frequency index freq_idx."""
        return self.H1_tilde if self.H1_tilde.ndim == 2 else self.H1_tilde[freq_idx]

    def omega_d_to_idx(self, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to omega_d value."""
        return np.argmin(np.abs(self.omega_d_values - omega_d))

    def amp_to_idx(self, amp: float, omega_d: float) -> np.ndarray[int]:
        """Return index corresponding to amplitude value.

        Because the drive amplitude can depend on the drive frequency, we also must pass
        the drive frequency here.
        """
        omega_d_idx = self.omega_d_to_idx(omega_d)
        return np.argmin(np.abs(self.drive_amplitudes[:, omega_d_idx] - amp))

    def hamiltonian(self, omega_d: float, amp: float, freq_idx: int = 0) -> dq.TimeQArray:
        """Return the modulated Hamiltonian in the rotating frame; i.e. the Hamiltonian we actually simulate."""
        H0_evals, H1_tilde = self.H0_evals_at(freq_idx), self.H1_tilde_at(freq_idx)

        def H(t):
            # diag(p) @ M @ diag(p)^dag == p[:, None] * M * conj(p)[None, :],
            # i.e. O(N^2) rather than O(N^3)
            p = jnp.exp(1j * H0_evals * t)
            return dq.asqarray(
                amp * jnp.cos(omega_d * t) * p[:, None] * H1_tilde * p.conj()[None, :]
            )
        return dq.timecallable(H)

    def bare_state_array(self) -> np.ndarray:
        """Return array of true bare states.

        Used to specify initial bare states and to compute average excitation
        number for the Blais branch analysis.
        """
        return np.identity(self.hilbert_dim)

    def mean_excitation(self, floquet_modes: jax.Array) -> jax.Array:
        """Mean excitation number of floquet modes, in the H0 eigenbasis.

        Each bare state is weighted by its excitation number. Can be overridden for
        models where another measure is more natural, e.g. for multi-mode systems.
        Written in jax.numpy, since the branch analysis calls it under jit.

        Parameters:
            floquet_modes: Shape `(num_omega_ds, num_modes, hilbert_dim)`.

        Returns:
            Mean excitation of each mode. Shape `(num_omega_ds, num_modes)`.
        """
        # The bare states are the basis states (bare_state_array is the identity), so the
        # overlaps are the components. Avoids an O(hilbert_dim^3) contraction.
        overlaps_sq = jnp.abs(jnp.swapaxes(floquet_modes, -1, -2)) ** 2

        # The excitation number of a bare state is its index if H0 was given in its
        # eigenbasis, and otherwise its rank in energy
        excitation = jnp.arange(self.hilbert_dim)
        if self.H0_evecs is not None:
            excitation = jnp.argsort(jnp.argsort(self.H0_evals, axis=-1), axis=-1)

        # sum over bare excitations weighted by excitation number
        return jnp.real(jnp.einsum("...ik,...i->...k", overlaps_sq, excitation))
