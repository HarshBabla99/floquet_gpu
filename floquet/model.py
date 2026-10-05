from __future__ import annotations

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt


class Model(eqx.Module):
    """Specify the model, including the Hamiltonian, drive strengths and frequencies.

    Can be subclassed to e.g. override the hamiltonian() method for a different (but
    still periodic!) Hamiltonian.

    Parameters:
        H0: Drift Hamiltonian, which must be diagonal and provided in units such that
            H0 can be passed directly to qutip.
        H1: Drive operator, which should be unitless (for instance the charge-number
            operator n of the transmon). It will be multiplied by a drive amplitude
            that we scan over from drive_parameters.drive_amplitudes.
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

        # diagonalize the drift Hamiltonian
        # NOTE: In our case H0 is diagonal!!
        # If not: 
        # evals, evecs = np.linalg.eigh(H0)
        # H1_tilde = evecs.conj().T @ H1 @ evecs
        self.H0_evals = jnp.diag(H0.to_jax())
        self.H1_tilde = H1.to_jax()
        self.hilbert_dim = H0.shape[-1]

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

    def hamiltonian(self, omega_d: float, amp: float) -> dq.TimeQArray:
        """Return the modulated Hamiltonian in the rotating frame; i.e. the Hamiltonian we actually simulate."""
        
        def H(t):
            # diag(p) @ M @ diag(p)^dag == p[:, None] * M * conj(p)[None, :], 
            # i.e. O(N^2) rather than O(N^3)
            p = jnp.exp(1j * self.H0_evals * t)
            return dq.asqarray(
                amp * jnp.cos(omega_d * t) * p[:, None] * self.H1_tilde * p.conj()[None, :]
            )
        return dq.timecallable(H)

    def bare_state_array(self) -> np.ndarray:
        """Return array of true bare states.

        Used to specify initial bare states and to compute average excitation
        number for the Blais branch analysis.
        """
        return np.identity(self.hilbert_dim)
