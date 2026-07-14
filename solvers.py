import jax.numpy as jnp
import dynamiqs as dq
import qutip as qt
import numpy as np

dq.set_precision('double')
dq.set_progress_meter(False)

N_steps = 128
rtol_atol = 1e-8
method = dq.method.Expm()
options = dq.Options(save_propagators=True, progress_meter=False, t0=0)

##
# Helpers
##
def propagator(H0, H1, A, omega_d):
    """Get the propagator given a drift Hamiltonian, drive Hamiltonian,
    drive amplitude, and drive frequency."""
    T = 2.0 * jnp.pi / omega_d
    dt = T / N_steps

    times = jnp.arange(N_steps + 1) * dt
    t_mid = (jnp.arange(N_steps) + 0.5) * dt

    H = dq.constant(H0) + dq.pwc(times, A * jnp.cos(omega_d * t_mid), H1)
    seprop_result = dq.sepropagator(H, [T], method=method, options=options)
    return seprop_result.final_propagator

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
def floquet_basic(H, omega_d):
    T = 2.0 * np.pi / omega_d
    fbasis = qt.FloquetBasis(H, T, options={'rtol': rtol_atol, 'atol': rtol_atol})
    f_modes_t = fbasis.mode(0.0, data=True).to_array()
    return fbasis.e_quasi, f_modes_t

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

def floquet_sambe(H0, H1, A, omega_d, N, dense=False):
    hilbert_dim = H0.dims[0]
    T = 2.0 * jnp.pi / omega_d
    num_blocks = 2 * N + 1
    
    # H0 in every block
    Q = dq.eye(num_blocks) & H0

    # Integers * omega_d ladder
    omega_d_multiples = dq.sparsedia_from_dict({0: omega_d*jnp.arange(-N, N + 1)})
    Q += omega_d_multiples & dq.eye_like(H0)

    # Drive coupling
    off_diag = jnp.ones(num_blocks-1, dtype=jnp.complex128)
    S = dq.sparsedia_from_dict({1: off_diag, -1:off_diag}, dims=(num_blocks,))
    Q += S & (0.5 * A * H1)

    # Make Hermitian
    Q = 0.5 * (Q + Q.dag())

    # Diagonalize
    if dense:
        w, V = Q.asdense()._eigh()
    else:
        w, V = Q._eigh()

    Vr = V.reshape(num_blocks, hilbert_dim, num_blocks * hilbert_dim)

    # Select the physical states of the central (m=0) block
    central_w = jnp.sum(jnp.abs(Vr[N]) ** 2, axis=0)
    perm = jnp.argsort(central_w)[-hilbert_dim:]

    evals = jnp.exp(-1j * w[perm] * T)

    # t=0 Floquet mode |u_k(0)> = sum_m phi_k^{(m)}  (sum over Fourier blocks)
    modes = jnp.sum(Vr[:, :, perm], axis=0)
    modes = modes / jnp.linalg.norm(modes, axis=0, keepdims=True)

    return evals, dq.asqarray(modes)