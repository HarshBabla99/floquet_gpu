import jax.numpy as jnp
import dynamiqs as dq
import qutip as qt
import numpy as np

dq.set_precision('double')
dq.set_progress_meter(False)

rtol_atol = 1e-8
dq_method = dq.method.Tsit5(rtol=rtol_atol, atol=rtol_atol)
qt_method = 'tsit5'

options = dq.Options(save_propagators=True, progress_meter=False, t0=0)

# Tighter reference tolerance
ref_rtol_atol = 1e-12

##############
# Reference (Off-the-shelf qutip)
##############
def qt_floquet(H, omega_d, tol=None):
    """Reference solver: QuTiP's FloquetBasis, integrated at `tol` (default ref_rtol_atol)."""
    tol = ref_rtol_atol if tol is None else tol
    T = 2.0 * np.pi / omega_d
    fbasis = qt.FloquetBasis(H, T, options={'rtol': tol, 'atol': tol,
                                            'method': qt_method})
    f_modes_t = fbasis.mode(0.0, data=True).to_array()
    return fbasis.e_quasi, f_modes_t, fbasis.U(T).full()

##############
# Propagators given a drift Hamiltonian, drive Hamiltonian, drive amplitude, and drive frequency.
##############

# Qutip propagators
def qt_lab_propagator(H0, H1, A, omega_d):
    """Get the propagator, by naively performing numerics in the lab frame.

    Returns a plain ndarray: np.array(Qobj) yields a 0-d object array, so the
    diagonalizers and nonunitarity() cannot consume a Qobj directly.
    """
    H = qt.QobjEvo([H0, [A * H1, lambda t: np.cos(omega_d * t)]])
    T = 2.0 * np.pi / omega_d
    U = qt.propagator(H, T, options={'rtol': rtol_atol, 'atol': rtol_atol, 'method': qt_method})
    return U.full()

def qt_ip_propagator(H0, H1, A, omega_d):
    """Get the propagator, but perform numerics in the interaction picture.

    Returns a plain ndarray, for the same reason as qt_lab_propagator.
    """

    # diagonalize the drift Hamiltonian
    # and transform H1 to the frame rotating with H0
    # NOTE: if H0 is diagonal, and H1 is sparse then
    # evals = H0.diag(), H1_tilde = H1
    evals, evecs = np.linalg.eigh(H0.full())
    H1_tilde = evecs.conj().T @ H1.full() @ evecs

    # time for a single period
    T = 2.0 * np.pi / omega_d

    # modulated Hamiltonian in the rotating frame
    def H(t):
        # diag(p) @ M @ diag(p)^dag == p[:, None] * M * conj(p)[None, :], 
        # i.e. O(N^2) rather than O(N^3)
        p = np.exp(1j * evals * t)
        return qt.Qobj(
            A * np.cos(omega_d * t) * p[:, None] * H1_tilde * p.conj()[None, :]
        )

    # interaction-picture propagator W(T), in the H0 eigenbasis
    U = qt.propagator(qt.QobjEvo(H), T,
                      options={'rtol': rtol_atol, 'atol': rtol_atol, 'method': qt_method})

    # undo the interaction picture, then rotate back to the lab basis
    U_tilde = U.full() * np.exp(-1j * evals * T)[:, None]
    return evecs @ U_tilde @ evecs.conj().T

# Dynamiqs propagators
def dq_lab_propagator(H0, H1, A, omega_d):
    """Get the propagator, by naively performing numerics in the lab frame. """
    H = dq.constant(H0) + dq.modulated(lambda t: A * jnp.cos(omega_d * t), H1)
    T = 2.0 * jnp.pi / omega_d
    ts = jnp.array([T])
    seprop_result = dq.sepropagator(H, ts, method=dq_method, options=options)
    return seprop_result.final_propagator

def dq_ip_propagator(H0, H1, A, omega_d):
    """Get the propagator, but perform numerics in the interaction picture. """

    # diagonalize the drift Hamiltonian
    # and transform H1 to the frame rotating with H0
    # NOTE: if H0 is diagonal, and H1 is sparse then
    # evals = H0.diag(), H1_tilde = H1
    evals, evecs = H0.asdense()._eigh()
    evecs = dq.asqarray(evecs)
    H1_tilde = (evecs.dag() @ H1 @ evecs).to_jax()

    # time for a single period
    T = 2.0 * jnp.pi / omega_d
    ts = jnp.array([T])

    # modulated Hamiltonian in the rotating frame
    def H(t):
        # diag(p) @ M @ diag(p)^dag == p[:, None] * M * conj(p)[None, :], 
        # i.e. O(N^2) rather than O(N^3)
        p = jnp.exp(1j * evals * t)
        return dq.asqarray(
            A * jnp.cos(omega_d * t) * p[:, None] * H1_tilde * p.conj()[None, :]
        )

    # interaction-picture propagator W(T), in the H0 eigenbasis
    seprop_result = dq.sepropagator(dq.timecallable(H), ts, method=dq_method, options=options)

    # undo the interaction picture, then rotate back to the lab basis
    U_tilde = seprop_result.final_propagator.elmul(jnp.exp(-1j * evals * T)[:, None])
    return evecs @ U_tilde @ evecs.dag()

##############
# Diagonalization
##############

# Qutip diag, via numpy
def qt_basic(U):
    evals, evecs = np.linalg.eig(np.array(U))
    return evals, evecs

def qt_cayley(U, phi=0):

    U = np.array(U)
    I = np.identity(U.shape[0])

    # Issue when U has an evalue of -1; causes a singularity in (I+U)^{-1}. 
    # Solution: rotate with a random phase. Eigenvectors are unchanged.
    W = np.exp(1j * phi) * U

    # construct the Hermitian matrix
    H = 1j * np.linalg.solve(I + W, I - W)
    H = 0.5 * (H + H.conj().T)

    # diagonalize hermitian
    _, V = np.linalg.eigh(H)

    # Recover evals of U: diag(V^dag U V) = (V^dag U V)_ii = \sum_j (V^*)_{ji} (U V)_{ji}
    lam = np.sum(np.conj(V) * (U @ V), axis=0)

    return lam, V

# Dynamiqs diag.
def dq_basic(U):
    # diagonalize the final propagator
    evals, evecs = U._eig()
    return evals, evecs

def dq_cayley(U, phi=0):
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

##############
# Polar project the numerically obtained propagator to the nearest unitary
# U = W P with W unitary and P positive-semidefinite; W = A B^dag from the SVD
# A S B^dag minimises ||U - W||_F over all unitaries. The exact propagator is unitary,
# so this can only move an integrated propagator toward the truth.
##############
def qt_polar_project(U):
    X = np.array(U)
    A, _, Bh = np.linalg.svd(X, full_matrices=False)
    return A @ Bh

def dq_polar_project(U):
    X = U.to_jax()
    A, _, Bh = jnp.linalg.svd(X, full_matrices=False)
    return dq.asqarray(A @ Bh)

##############
# Post processing
##############
def qt_post_process(evals, evecs, omega_d, quasienergies=None):

    # extract quasienergies (minus sign / divide by T for e^{-i eps T}).
    # The reference (FloquetBasis) supplies them directly; everything else passes evals.
    if quasienergies is None:
        T = 2.0 * np.pi / omega_d
        quasienergies = -np.angle(evals) / T

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

def dq_post_process(evals, evecs, omega_d):
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