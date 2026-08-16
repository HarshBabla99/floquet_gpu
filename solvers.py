import jax.numpy as jnp
import dynamiqs as dq
import qutip as qt
import numpy as np

dq.set_precision('double')
dq.set_progress_meter(False)

rtol_atol = 1e-8
method = dq.method.Tsit5(rtol=rtol_atol, atol=rtol_atol)
# QuTiP's own Tsitouras 5/4 implementation, so the reference uses the same
# order-5 adaptive RK as dynamiqs. QuTiP's default ('adams', a variable-order
# multistep) is ~35x less accurate at this tolerance once ||H||*T is large,
# which made qerr measure the reference's error rather than the solvers'.
qutip_method = 'tsit5'
#method = dq.method.Euler(dt=1e-4)
options = dq.Options(save_propagators=True, progress_meter=False, t0=0)

# Tighter reference tolerance
ref_rtol_atol = 1e-12

##
# Helpers
##
def propagator(H0, H1, A, omega_d):
    """Get the propagator given a drift Hamiltonian, drive Hamiltonian,
    drive amplitude, and drive frequency."""
    H = dq.constant(H0) + dq.modulated(lambda t: A * jnp.cos(omega_d * t), H1)
    T = 2.0 * jnp.pi / omega_d
    ts = jnp.array([T])
    seprop_result = dq.sepropagator(H, ts, method=method, options=options)
    return seprop_result.final_propagator

def ip_propagator(H0, H1, A, omega_d):
    """Get the propagator, but perform numerics in the interaction picture. """

    # diagonalize the drift Hamiltonian
    # and transform H1 to the frame rotating with H0
    # NOTE: if H0 is diagonal, and H1 is sparse then
    # evals = H0.diag(), H1_tilde = H1
    evals, evecs = H0.asdense()._eigh()
    evecs = dq.asqarray(evecs)
    H1_tilde = evecs.dag() @ H1 @ evecs

    # time for a single period
    T = 2.0 * jnp.pi / omega_d
    ts = jnp.array([T])

    # modulated Hamiltonian in the rotating frame
    def H(t):
        phases = dq.sparsedia_from_dict({0: jnp.exp(1j * evals * t)})
        return A * jnp.cos(omega_d * t) * (phases @ H1_tilde @ phases.dag())

    # interaction-picture propagator W(T), in the H0 eigenbasis
    seprop_result = dq.sepropagator(dq.timecallable(H), ts, method=method, options=options)

    # undo the interaction picture, then rotate back to the lab basis
    U_tilde = seprop_result.final_propagator.elmul(jnp.exp(-1j * evals * T)[:, None])
    return evecs @ U_tilde @ evecs.dag()

def nonunitarity(U):
    """Compute deviation of the integrated propagator from unitarity.
    Expected to grow linearly in ||H||*T. 
    """
    U = U.to_jax() if hasattr(U, 'to_jax') else jnp.asarray(U)
    D = U.conj().T @ U - jnp.eye(U.shape[-1], dtype=U.dtype)
    return float(jnp.abs(D).max())

def polar_project(U):
    """Project to nearest unitary matrix to U.

    U = W P with W unitary and P positive-semidefinite; W = A B^dag from the SVD
    A S B^dag minimises ||U - W||_F over all unitaries. The exact propagator is unitary,
    so this can only move an integrated propagator toward the truth.
    """
    X = U.to_jax() if hasattr(U, 'to_jax') else jnp.asarray(U)
    A, _, Bh = jnp.linalg.svd(X, full_matrices=False)
    return dq.asqarray(A @ Bh)

def hamiltonian_norm(X):
    """Spectral norms of a Hermitian operator.

    The spectral norm ||H||_2 (= largest |eigenvalue|) is the physically meaningful one:
    ||H0||_2 * T is the phase accumulated over one drive period, which sets the step count
    and hence the accuracy achievable at a given ODE tolerance.

    A purely diagonal SparseDIAQArray is handled in O(N) straight from its stored diagonal,
    so recording this neither densifies the operator nor inflates the peak-RSS memory metric.
    Dense operators fall back to eigvalsh, which is O(N^3) and is the dominant cost here.
    """
    if isinstance(X, dq.SparseDIAQArray) and tuple(X.offsets) == (0,):
        diag = X.diags[0]
        return float(jnp.abs(diag).max())

    A = X.to_jax() if hasattr(X, 'to_jax') else jnp.asarray(X)
    return float(jnp.abs(jnp.linalg.eigvalsh(A)).max())

def post_process_qutip(quasienergies, evecs, omega_d):

    # fold into the first Brillouin zone (-pi/T, pi/T]
    quasienergies = np.mod(quasienergies + 0.5 * omega_d, omega_d) - 0.5 * omega_d

    # remove the global phase on the maximum-magnitude component of each mode
    pivot = np.argmax(np.abs(evecs), axis=0)
    pv = np.take_along_axis(evecs, pivot[None, :], axis=0)[0]
    phase = pv / np.abs(pv)
    evecs = evecs * np.conj(phase)[None, :]

    # sort by quasienergy
    perm = np.argsort(quasienergies)
    return quasienergies[perm], evecs.T[perm]

def post_process(evals, evecs, omega_d):
    T = 2.0 * jnp.pi / omega_d

    # extract quasienergies (minus sign / divide by T for e^{-i eps T})
    quasienergies = -jnp.angle(evals) / T
    # fold into the first Brillouin zone (-pi/T, pi/T]
    quasienergies = jnp.mod(quasienergies + 0.5 * omega_d, omega_d) - 0.5 * omega_d

    # remove the global phase on the maximum-magnitude component of each mode
    evecs = evecs.to_jax()
    pivot = jnp.argmax(jnp.abs(evecs), axis=0)
    pv = jnp.take_along_axis(evecs, pivot[None, :], axis=0)[0]
    phase = pv / jnp.abs(pv)
    evecs = evecs * jnp.conj(phase)[None, :]

    # sort by quasienergy
    perm = jnp.argsort(quasienergies)
    return quasienergies[perm], dq.asqarray(evecs.T[perm])

##
# Solvers 
##
def floquet_basic(H, omega_d, tol=None):
    """Reference solver: QuTiP's FloquetBasis, integrated at `tol` (default ref_rtol_atol)."""
    tol = ref_rtol_atol if tol is None else tol
    T = 2.0 * np.pi / omega_d
    fbasis = qt.FloquetBasis(H, T, options={'rtol': tol, 'atol': tol,
                                            'method': qutip_method})
    f_modes_t = fbasis.mode(0.0, data=True).to_array()
    return fbasis.e_quasi, f_modes_t, fbasis.U(T).full()

def floquet_dq_basic(U):
    # diagonalize the final propagator
    evals, evecs = U._eig()
    return evals, evecs

def floquet_cayley(U, phi=0):
    I = dq.eye_like(U)

    # turn both into jax.numpy arrays
    I = I.to_jax()
    U = U.to_jax()

    # Issue when U has an evalue of -1; causes a singularity in (I+U)^{-1}. 
    # Solution: rotate with a random phase. Eigenvectors are unchanged.
    W = jnp.exp(1j * phi) * U

    # construct the Hermitian matrix
    H = 1j * jnp.linalg.solve(I + W, I - W)
    H = 0.5 * (H + H.conj().T)

    # diagonalize hermitian
    _, V = jnp.linalg.eigh(H)

    # Recover evals of U: diag(V^dag U V) = (V^dag U V)_ii = \sum_j (V^*)_{ji} (U V)_{ji}
    lam = jnp.sum(jnp.conj(V) * (U @ V), axis=0)

    return lam, dq.asqarray(V)