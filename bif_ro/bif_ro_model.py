import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import qutip as qt
import scqubits as scq
from floquet.floquet import FloquetAnalysis
from floquet.model import Model
from jax.lax.linalg import eigh as lax_eigh, EighImplementation
from scipy.optimize import linear_sum_assignment, root

####################################################################################################
# Transmon
def tmon(EJ, EC, ng, ncut, transmon_dim, **_):
    """
    Returns:
        H0 = transmon Hamiltonian
        n = transmon charge operator
        eye = transmon identity operator
    """
    tmon = scq.Transmon(EJ = EJ, EC = EC,
                        ng = ng, ncut = ncut,
                        truncated_dim = transmon_dim)

    # Get operators in the eigenbasis
    H0 = qt.Qobj(tmon.hamiltonian(energy_esys=True))
    n = qt.Qobj(tmon.n_operator(energy_esys=True))
    eye = qt.qeye(transmon_dim)

    # Convention: EJ and EC are in 2pi*GHz, so the Hamiltonian is also in 2pi*GHz
    return (H0 - H0[0,0]), n, eye

# Kerr oscillator
def kerr_cat(omega_r, K, resonator_dim, **_):
    """
    Returns:
        H0 = resonator Hamiltonian
        H1 = resonator drive operator (squeezing)
        a = resonator annihilation operator
        adag = resonator creation operator
        eye = resonator identity operator
    """
    a = qt.destroy(resonator_dim)
    adag = a.dag()
    eye = qt.qeye(resonator_dim)

    # Static Hamiltonian
    H0 = omega_r * adag * a
    H0 += (-K) * adag * adag * a * a

    # Charge drive; time in 1/GHz = ns
    # Will be multiplied by the drive amplitude and an oscillating factor
    H1 = (adag * adag) + (a * a)

    # Convention: omega_r and K are in 2pi*GHz, so units are:
    # H0 => 2pi*GHz, H1 => dimensionless
    return H0, H1, a, adag, eye

# coupled system
def coupled_system(H0_t, n_t, eye_t, H0_r, H1_r, a_r, adag_r, eye_r, g):

    # static Hamiltonian
    H0 = qt.tensor(H0_t, eye_r) + qt.tensor(eye_t, H0_r)
    H0 += 1j * g * qt.tensor(n_t, adag_r - a_r)

    # Drive term
    # Units: dimensionless (will get multiplied by 2pi * amp)
    H1 = qt.tensor(eye_t, H1_r)

    return H0, H1

def eigh_with_retry(H: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Batched eigh on the device.

    On GPU, the default eigensolver very rarely fails without warning and returns NaNs
    (see floquet.compute_floquet_grid). The failed matrices are recomputed with Jacobi.
    """
    evals, evecs = jnp.linalg.eigh(H)
    ok = jnp.all(jnp.isfinite(evecs), axis=(-2, -1)) & jnp.all(jnp.isfinite(evals), axis=-1)
    for idx in np.argwhere(~np.asarray(ok)):
        idx = tuple(idx)
        # lax.linalg.eigh returns (evecs, evals)
        v, e = lax_eigh(H[idx], implementation=EighImplementation.JACOBI)
        evals, evecs = evals.at[idx].set(e), evecs.at[idx].set(v)
    if not (jnp.all(jnp.isfinite(evals)) and jnp.all(jnp.isfinite(evecs))):
        raise FloatingPointError("eigh still non-finite after the Jacobi retry")
    return evals, evecs

####################################################################################################
class BifurcationROModel(Model):
    """Floquet model of a transmon coupled to a Kerr-cat resonator, for a range of drive
    frequencies. Built by BifurcationROSystem.model().

    The operators are in the product state basis |i_t, j_r>, with basis index
    i * resonator_dim + j. The frequency-dependent H0 is diagonalized and the dressed
    states are labelled by these product states, so the state indices are product-state
    indices.
    """

    # (transmon_dim, resonator_dim)
    subsystem_dims: tuple = eqx.field(static=True)

    def __init__(self, H0, H1, omega_d_values, drive_amplitudes, subsystem_dims):
        """Same as Model.__init__, but diagonalizes H0 on the device.

        Parameters:
            H0: Static Hamiltonians in the product basis. Shape (num_omega_ds, n, n).
            H1: Drive operator in the product basis. Shape (n, n).
            omega_d_values: Drive frequencies. Shape (num_omega_ds,).
            drive_amplitudes: Shape (num_amps, num_omega_ds).
            subsystem_dims: (transmon_dim, resonator_dim).
        """
        self.omega_d_values = np.asarray(omega_d_values)
        self.drive_amplitudes = np.asarray(drive_amplitudes)
        assert self.drive_amplitudes.shape[1] == len(self.omega_d_values)

        # The simulation is performed in the eigenbasis of the drift Hamiltonian
        self.H0_evals, self.H0_evecs = self.diagonalize(jnp.asarray(H0))
        self.H1_tilde = (
            jnp.conj(jnp.swapaxes(self.H0_evecs, -1, -2)) @ jnp.asarray(H1) @ self.H0_evecs
        )
        self.hilbert_dim = H0.shape[-1]
        self.subsystem_dims = tuple(subsystem_dims)

    @staticmethod
    def diagonalize(H0: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Model.diagonalize on the device.

        The eigenvectors are labelled by the basis state they overlap most, which is the
        argmax unless two basis states pick the same eigenvector. Only then is scipy's
        assignment used, on the host. The phase is fixed such that this overlap is real
        and positive.
        """
        evals, evecs = eigh_with_retry(H0)

        # perm[..., k] is the eigenvector assigned to basis state k
        perm = np.array(jnp.argmax(jnp.abs(evecs), axis=-1))
        for idx in zip(*np.nonzero(FloquetAnalysis._clashes(perm))):
            perm[idx] = linear_sum_assignment(-np.abs(np.asarray(evecs[idx])))[1]
        perm = jnp.asarray(perm)
        evals = jnp.take_along_axis(evals, perm, axis=-1)
        evecs = jnp.take_along_axis(evecs, perm[..., None, :], axis=-1)

        overlaps = jnp.einsum("...kk->...k", evecs)
        evecs = evecs * jnp.conj(overlaps / jnp.abs(overlaps))[..., None, :]
        return evals, evecs

    def mean_excitation(self, floquet_modes: jax.Array) -> jax.Array:
        r"""Mean excitation number of each subsystem, and of the composite system.

        For a state $|\psi> = \sum_{ij} c_{ij} |i_t, j_r>$, the transmon's mean
        excitation is $\sum_i i * \sum_j |c_{ij}|^2$, which is just a tensor contraction,
        i.e. the excitation of the reduced density matrix without building it.
        Written in jax.numpy, since the branch analysis calls it under jit.

        Parameters:
            floquet_modes: In the dressed basis. Shape: (num_omega_ds, num_modes, hilbert_dim).

        Returns:
            Mean excitation of the transmon, resonator, and composite system (the
            dressed level, as in Model.mean_excitation).
                Shape: (num_omega_ds, len(subsystem_dims) + 1, num_modes).
        """
        # Floquet modes as kets in the product basis: psi = evecs @ c
        states = floquet_modes @ jnp.swapaxes(self.H0_evecs, -1, -2)

        # Split the product-basis axis into one axis per subsystem
        probs = jnp.abs(states.reshape(*states.shape[:-1], *self.subsystem_dims)) ** 2
        subsys_axes = tuple(range(states.ndim - 1, probs.ndim))
        probs = probs / probs.sum(axis=subsys_axes, keepdims=True)

        mean_excitation = []
        for subsys_axis, subsys_dim in zip(subsys_axes, self.subsystem_dims):
            # Marginalize over every subsystem except this one
            traced_out = tuple(axis for axis in subsys_axes if axis != subsys_axis)
            mean_excitation.append(probs.sum(axis=traced_out) @ jnp.arange(subsys_dim))

        mean_excitation.append(super().mean_excitation(floquet_modes))
        return jnp.stack(mean_excitation, axis=-2)


class BifurcationROSystem:
    r"""Specify the system, including the Hamiltonian, drive strengths and frequencies.

    The Hamiltonian is specified as
    $$
    H = H0a + omega_r*H0b + amp*H1*cos(omega_d*t)
    $$
    Therefore:
    - H0a should have units of $2 \pi$ GHz
    - H0b and H1 should be unitless
    - omega_r, omega_d, and amp should have units of $2 \pi$ GHz

    Importantly:
    All operators (H0a, H0b, H1) should be in the product state basis!

    The Floquet simulation itself runs on the models returned by model().

    Parameters:
        see params.py
    """

    def __init__(
        self,
        **params: dict,
    ):

        # Validate
        for k in ['EJ', 'EC', 'ng', 'ncut', 'transmon_dim', 'K', 'delta_const', 'resonator_dim', 'omega_r_values', 'photon_nums', 'chi']:
            assert k in params.keys(), f"Parameter '{k}' is required"

            if k in ['omega_r_values', 'photon_nums']:
                assert isinstance(params[k], (list, np.ndarray)), f"Parameter '{k}' must be a list or numpy array"
                params[k] = np.array(params[k])

            elif k in ['transmon_dim', 'resonator_dim']:
                assert isinstance(params[k], int), f"Parameter '{k}' must be an integer"

            else:
                assert isinstance(params[k], (int, float)), f"Parameter '{k}' must be a number"

        # Store parameters
        # Note: omega_d = 2 * omega_r always, because we optimize the params s.t.
        # dressed resonator freq is omega_r
        self.omega_r_values = params['omega_r_values']
        self.omega_d_values = 2. * np.array(self.omega_r_values) + params['delta_const']

        # Hamiltonian term: drive_amp * (adag^2 + a^2) * cos(omega_d t)
        # cosine contributes 1/2 in the RWA, so eps_2 = amp/2 = K*nbar forms a cat with nbar photons
        # Corresponding transmon AC Stark shift is stark_shifts = chi*nbar
        self.drive_amplitudes = np.tile(2. * params['K'] * params['photon_nums'][:, None],
                                        (1, len(self.omega_r_values)))
        self.stark_shifts = params['chi'] * params['photon_nums']
        self.params = params

        # Hilbert dims
        self.hilbert_dim = [
                self.params['transmon_dim'],
                self.params['resonator_dim'],
                self.params['transmon_dim'] * self.params['resonator_dim']
            ]

        # Cache the transmon operators. They depend on neither the resonator
        # frequency nor the drive, so rebuilding them (a scqubits diagonalization)
        # for every resonator frequency would repeat the same work.
        self.tmon_ops = tmon(**self.params)

        # Optimize for the system parameters
        (
            self.omega_r_optimized,
            self.kerr_optimized,
            self.g_optimized
        ) = self.optimize_system_parameters()

        # The drive operator eye_t (x) (adag^2 + a^2) does not depend on
        # (omega_r, K, g), so it is the same for every resonator frequency.
        _, self.H1_drive = coupled_system(
            *self.tmon_ops,
            *kerr_cat(self.omega_r_optimized[0], self.kerr_optimized[0], self.params['resonator_dim']),
            self.g_optimized[0],
        )

        # The static Hamiltonian is affine in (omega_r, K, g):
        # H0 = A + omega_r * B + K * C + g * D, see static_hamiltonian().
        # Store A, B, C, D on the device, so model() builds H0 for a batch of frequencies there.
        dim_r = self.params['resonator_dim']
        H0_t, n_t, eye_t = self.tmon_ops
        no_tmon = (0 * H0_t, n_t, eye_t)
        self.H0_terms = [
            jnp.asarray(coupled_system(*ops, *kerr_cat(omega_r, kerr, dim_r), g)[0].full())
            for ops, (omega_r, kerr, g) in zip(
                (self.tmon_ops, no_tmon, no_tmon, no_tmon),
                ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
            )
        ]

        # Diagonalize the undriven Hamiltonian for all resonator frequencies.
        # The eigenvectors are not stored (num_omega_r x hilbert_dim^2 is too large); the
        # models re-diagonalize the static Hamiltonians of their batch of frequencies.
        self.undriven_evals = np.zeros((
                                            len(self.omega_r_values),
                                            self.hilbert_dim[-1]
                                       ), dtype=float)

        self.assignment_idxs = np.zeros((
                                            len(self.omega_r_values),
                                            *self.hilbert_dim[:-1]
                                        ), dtype=int)

        # dressed parameters: omega_q, omega_r, alpha_q, Kerr, chi
        self.dressed_parameters = np.zeros((
                                            len(self.omega_r_values), 5
                                        ), dtype=float)

        for omega_r_idx, _ in enumerate(self.omega_r_values):
            evals, evecs = self.diagonalize(self.static_hamiltonian(omega_r_idx))
            self.undriven_evals[omega_r_idx, :] = evals
            self.assignment_idxs[omega_r_idx] = self.get_assignment_idxs(evecs)
            self.dressed_parameters[omega_r_idx] = self.get_dressed_parameters(
                evals, self.assignment_idxs[omega_r_idx]
            )


    ################
    # Construct the system
    ################
    def optimize_system_parameters(self):
        """Solve for the bare (omega_r, Kerr, g) reproducing the target dressed
        parameters at every omega_r. Three unknowns and three targets, so this is an exact root-finding problem.
        """
        # The transmon, built in __init__, fixes ω_t and α_t via EJ and EC
        H0_t, n_t, eye_t = self.tmon_ops
        omega_q = np.real(H0_t[1, 1] - H0_t[0, 0])
        alpha_q = -np.real(H0_t[2, 2] - H0_t[1, 1] - omega_q)

        # Find a omega_r_bare, coupling strength and Kerr for each drive frequency.
        # Note: chi and Kerr are both positive here; get_dressed_parameters() returns
        # them as positive shifts (H = -K adag^2 a^2, omega_r^e = omega_r^g - chi).
        # Start with the following guesses
        omega_r_bare = self.omega_r_values
        delta_qr = np.real(omega_q - omega_r_bare)
        g_values = np.sqrt(self.params['chi'] * delta_qr * (delta_qr - alpha_q)/(2 * alpha_q))
        # Induced resonator self-Kerr = 2*sqrt(K_q * K_r), i.e. K_induced = chi^2/(4*alpha_q)
        K_induced = self.params['chi']**2 / (4 * alpha_q)
        K_additional = self.params['K'] - K_induced

        # Residuals are relative, so all three are O(1) regardless of whether the
        # target is a GHz frequency or a kHz nonlinearity
        def residuals(x, target_omega_r, target_Kerr, target_chi):
            omega_r, kerr, g = x
            H0, _ = coupled_system(
                H0_t, n_t, eye_t,
                *kerr_cat(omega_r, kerr, self.params['resonator_dim']),
                g
            )
            evals, evecs = self.diagonalize(H0)
            assgn_idxs = self.get_assignment_idxs(evecs)
            _, derived_omega_r,_, derived_Kerr, derived_chi = self.get_dressed_parameters(
                                                                                evals, assgn_idxs
                                                                            )
            return [
                1. - derived_omega_r / target_omega_r,
                1. - derived_Kerr / target_Kerr,
                1. - derived_chi / target_chi,
            ]

        # For each drive frequency, solve for the device parameters
        omega_r_optimized = np.zeros_like(self.omega_r_values)
        kerr_optimized = np.zeros_like(self.omega_r_values)
        g_optimized = np.zeros_like(self.omega_r_values)

        # Initial guess is the analytical estimate above
        _guess_params = [omega_r_bare[0], K_additional, g_values[0]]

        for idx, target_omega_r in enumerate(self.omega_r_values):
            if idx % 20 == 0:
                print(f"Optimizing for freq_r (idx = {idx}): {target_omega_r/(2*np.pi):.3f} GHz", flush=True)

            _residuals = lambda x : residuals(x,
                                              target_omega_r,
                                              self.params['K'],
                                              self.params['chi']
                                             )

            # try guessing with the previous solution first
            opt_result = root(_residuals, _guess_params)

            # If that fails, retry from a nudged guess.
            if not opt_result.success:
                nudged_guess = [*_guess_params[:-1], 0.95 * _guess_params[-1]]
                opt_result = root(_residuals, nudged_guess)

            if not opt_result.success:
                raise RuntimeError(
                                    f"Parameter solve failed at omega_r/2pi = "
                                    f"{target_omega_r/(2*np.pi):.3f} GHz: {opt_result.message} "
                                    f"(residuals = {opt_result.fun})"
                                )

            # set the guess params to the solution for the next iteration
            _guess_params = opt_result.x

            # also store the solution
            omega_r_optimized[idx], kerr_optimized[idx], g_optimized[idx] = opt_result.x

        return omega_r_optimized, kerr_optimized, g_optimized

    def static_hamiltonian(self, omega_r_idx: int) -> qt.Qobj:
        """The undriven coupled Hamiltonian (product state basis) at one resonator frequency."""
        H0, _ = coupled_system(
            *self.tmon_ops,
            *kerr_cat(
                self.omega_r_optimized[omega_r_idx],
                self.kerr_optimized[omega_r_idx],
                self.params['resonator_dim'],
            ),
            self.g_optimized[omega_r_idx],
        )
        return H0

    def model(self, freq_slice: slice = slice(None)) -> BifurcationROModel:
        """Floquet model for a range of the drive frequencies (one batch), built on the device."""
        A, B, C, D = self.H0_terms
        omega_r, kerr, g = (
            jnp.asarray(x[freq_slice])[:, None, None]
            for x in (self.omega_r_optimized, self.kerr_optimized, self.g_optimized)
        )
        H0 = A + omega_r * B + kerr * C + g * D
        return BifurcationROModel(
            H0,
            self.H1_drive.full(),
            omega_d_values=self.omega_d_values[freq_slice],
            drive_amplitudes=self.drive_amplitudes[:, freq_slice],
            subsystem_dims=self.hilbert_dim[:-1],
        )

    def diagonalize(self, H0: qt.Qobj) -> tuple[np.ndarray, np.ndarray]:
        # Diagonalize the static Hamiltonian
        # evecs.shape = (num_states, hilbert_dim)
        evals, evecs = np.linalg.eigh(H0.full())
        evecs = evecs.T

        # eigh results in a random global phase for each eigvec.
        # it jumps around for adjacent omega_r, which is bad for the displaced-state fit.
        dominant = np.argmax(np.abs(evecs), axis=-1)
        phases = np.angle(evecs[np.arange(len(dominant)), dominant])
        evecs *= np.exp(-1j * phases)[:, None]

        return evals, evecs

    def get_dressed_parameters(self, evals : np.ndarray, assgn_idxs : np.ndarray
                          ) -> tuple[float, float, float]:
        """Diagonalize a static coupled Hamiltonian and extract the dressed
        resonator frequency, Kerr, and chi.
        """
        # Get the dressed parameters
        omega_q = evals[assgn_idxs[1,0]] - evals[assgn_idxs[0,0]]
        omega_r = evals[assgn_idxs[0,1]] - evals[assgn_idxs[0,0]]

        alpha_q = -(
              (evals[assgn_idxs[2,0]] - evals[assgn_idxs[1,0]])
            - (evals[assgn_idxs[1,0]] - evals[assgn_idxs[0,0]])
        )
        # The ladder anharmonicity -(E02 - 2*E01 + E00) equals 2K for
        # H = -K adag^2 a^2, hence the factor of 1/2
        Kerr = -(
              (evals[assgn_idxs[0,2]] - evals[assgn_idxs[0,1]])
            - (evals[assgn_idxs[0,1]] - evals[assgn_idxs[0,0]])
        ) / 2.

        chi = -(
              (evals[assgn_idxs[1,1]] - evals[assgn_idxs[0,0]])
            - (evals[assgn_idxs[1,0]] - evals[assgn_idxs[0,0]])
            - (evals[assgn_idxs[0,1]] - evals[assgn_idxs[0,0]])
        )

        return omega_q, omega_r, alpha_q, Kerr, chi

    def get_assignment_idxs(self, evecs) -> np.ndarray:
        r"""Generate the assignment indices for the bare states.

        This is done by calculating the overlap of the bare states with the product
        states, and assigning each product state to the bare state with which it has
        the largest overlap.
        """
        # Since we're working in the product state basis, the list of product states is just
        # the identity reshaped.
        # So, overlap of evec[k] with $|i_t, j_r>$ is just |evecs[k, i*resonator_dim + j]|
        # Simply split the basis axis into (i, j).
        ovlps = np.moveaxis(np.abs(evecs), -1, -2)
        ovlps = ovlps.reshape(*evecs.shape[:-2], *self.hilbert_dim[:-1], -1)
        return np.argmax(ovlps, axis=-1).astype(int)
