import numpy as np
import qutip as qt
import scqubits as scq
from floquet.model import Model
from scipy.optimize import root

####################################################################################################
# Helper
def Qobj_to_np(obj : qt.Qobj | list[qt.Qobj] | np.ndarray, isket: bool = False):
    """Convert a (list of) Qobjs to a numpy array. """
    
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            obj = obj.tolist()
        else:
            return obj
    
    # A single object
    if isinstance(obj, qt.Qobj):
        if isket:
            return np.array(obj.unit().tidyup().full(squeeze=True), dtype=complex)
        else:
            return np.array(obj.tidyup().full(squeeze=True), dtype=complex)
    
    # A list (of lists) of objects
    elif isinstance(obj, list):
        return np.array([Qobj_to_np(sub_obj, isket) for sub_obj in obj])
        
    else:
        raise ValueError(f"obj can only be an instance of qutip.Qobj or a list of Qobj.")

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

####################################################################################################
class BifurcationROModel(Model):
    r"""Specify the model, including the Hamiltonian, drive strengths and frequencies.

    Can be subclassed to e.g. override the hamiltonian() method for a different (but
    still periodic!) Hamiltonian.

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
        # inside hamiltonian() would repeat the same work once per drive amplitude.
        self.tmon_ops = tmon(**self.params)

        # Optimize for the system parameters
        (
            self.omega_r_optimized,
            self.kerr_optimized,
            self.g_optimized
        ) = self.optimize_system_parameters()

        # Cache the coupled operators, one static Hamiltonian per resonator
        # frequency. The drive operator eye_t (x) (adag^2 + a^2) does not depend on
        # (omega_r, K, g), so every iteration assigns the same H1.
        self.H0_static = []
        for omega_r_idx, _ in enumerate(self.omega_r_values):
            H0, self.H1_drive = coupled_system(
                *self.tmon_ops,
                *kerr_cat(
                    self.omega_r_optimized[omega_r_idx],
                    self.kerr_optimized[omega_r_idx],
                    self.params['resonator_dim'],
                ),
                self.g_optimized[omega_r_idx],
            )
            self.H0_static.append(H0)

        # Diagonalize the undriven Hamiltonian for all resonator frequencies
        self.undriven_evals = np.zeros((
                                            len(self.omega_r_values), 
                                            self.hilbert_dim[-1]
                                       ), dtype=float)
                                        
        self.undriven_evecs = np.zeros((
                                            len(self.omega_r_values), 
                                            self.hilbert_dim[-1], 
                                            self.hilbert_dim[-1]
                                       ), dtype=complex)

        for omega_r_idx, _ in enumerate(self.omega_r_values):
            evals, evecs = self.diagonalize(self.H0_static[omega_r_idx])
            self.undriven_evals[omega_r_idx, :]   = evals
            self.undriven_evecs[omega_r_idx, ...] = evecs

        self.assignment_idxs = self.get_assignment_idxs(self.undriven_evecs)

        # dressed parameters: omega_q, omega_r, alpha_q, Kerr, chi
        self.dressed_parameters = np.zeros((
                                            len(self.omega_r_values), 5
                                        ), dtype=float)

        for omega_r_idx, _ in enumerate(self.omega_r_values):
            self.dressed_parameters[omega_r_idx] = self.get_dressed_parameters(
                self.undriven_evals[omega_r_idx], self.assignment_idxs[omega_r_idx]
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
                print(f"Optimizing for freq_r (idx = {idx}): {target_omega_r/(2*np.pi):.3f} GHz")

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

    def hamiltonian(self,
                    omega_d_idx : int,
                    amp_idx : int | None,
                   ) -> qt.QobjEvo | qt.Qobj:
        """Return the Hamiltonian (in the product state basis) we actually simulate.

        The operators are built once in __init__; this only applies the scalars.
        """
        # If no amp_idx provided, get the undriven Hamiltonian
        if amp_idx is None:
            return self.H0_static[omega_d_idx]

        # Else, get the complete time-dependent driven Hamiltonian
        omega_d = self.omega_d_values[omega_d_idx]
        amp = self.drive_amplitudes[amp_idx, omega_d_idx]
        return qt.QobjEvo([
            self.H0_static[omega_d_idx],
            [amp * self.H1_drive, lambda t: np.cos(omega_d * t)],
        ])

    def bare_state_array(self) -> np.ndarray:
        """Return array of true bare states.

        Used to specify initial bare states and to compute average excitation
        number for the Blais branch analysis.
        """
        return self.undriven_evecs

    ################
    # Saving
    ################
    # don't save the qutip operators.
    SKIP_ATTRS = ('tmon_ops', 'H0_static', 'H1_drive')

    # Saved as their own .npy rather than pickled into params_<mode>.npy
    # Need to stream them as memap.
    STREAMED_ATTRS = ('undriven_evecs',)

    def data_to_save(self) -> dict:
        """All the data the model in the model, except SKIP_ATTRS and STREAMED_ATTRS.
        Written to params_<mode>.npy, by FloquetAnalysis. 
        """
        skip = (*self.SKIP_ATTRS, *self.STREAMED_ATTRS)
        data = {key: val for key, val in vars(self).items() if key not in skip}
        nested_params = data.pop('params')
        return data | nested_params

    def arrays_to_save(self, omega_r_idx: int = 0) -> dict:
        """The STREAMED_ATTRS for one drive frequency, as plain arrays.
        Freq axis is dropped (because it's restored by merge.py).
        """
        return {
            name: getattr(self, name)[omega_r_idx] for name in self.STREAMED_ATTRS
        }

    def ptrace(self, states: np.ndarray, keep_dim: int | list[int]):
        """Take the partial trace of an array of quantum states. 
        Inspired by qutip's ptrace.

        Parameters:
            states: a numpy array of states. 
                Assumes the states are kets of length self.hilbert_dim[-1]. 
                Shape: (..., self.hilbert_dim[-1]).
            keep_dim: The index of the subsystem(s) to keep after the partial trace.
        """
        if isinstance(keep_dim, int):
            keep_dim = [keep_dim]

        dims = [self.hilbert_dim[:-1], [1]*len(self.hilbert_dim[:-1])]
        ptraced_states_qt = [
            qt.Qobj(psi, dims=dims).unit().tidyup().ptrace(keep_dim)
            for psi in states
        ]
        return Qobj_to_np(ptraced_states_qt)

    def subsystem_mean_excitation(self, states: np.ndarray) -> np.ndarray:
        """Mean excitation number of each subsystem, for an array of kets.
        If ptrace forms a subsystem density matrix rho, then this function computes
        $\sum_n n \rho_{n,n}$, but without building the reduced density matrices.

        How? Well, for a state $|\psi> = \sum_{ij} c_{ij} |i_t, j_r>$, the transmon's mean
        excitation is $\sum_i i * \sum_j |c_{ij}|^2$, which is just a tensor contraction. 

        Parameters:
            states: Kets in the product basis. Shape: (num_states, hilbert_dim[-1]).

        Returns:
            Mean excitation of each subsystem.
                Shape: (len(self.hilbert_dim) - 1, num_states).
        """
        # Split the product-basis axis into one axis per subsystem
        probs = np.abs(states.reshape(len(states), *self.hilbert_dim[:-1])) ** 2
        probs /= probs.sum(axis=tuple(range(1, probs.ndim)), keepdims=True)

        mean_excitation = np.zeros((len(self.hilbert_dim) - 1, len(states)))
        for subsys_idx, subsys_dim in enumerate(self.hilbert_dim[:-1]):
            # Marginalize over every subsystem except this one
            traced_out = tuple(
                axis for axis in range(1, probs.ndim) if axis != subsys_idx + 1
            )
            mean_excitation[subsys_idx] = probs.sum(axis=traced_out) @ np.arange(subsys_dim)

        return mean_excitation

    ################
    # Indexing methods
    ################
    def omega_r_to_idx(self, omega_r: float) -> np.ndarray[int]:
        """Return index corresponding to omega_r value."""
        return np.argmin(np.abs(self.omega_r_values - omega_r))

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
