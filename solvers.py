import jax.numpy as jnp
import dynamiqs as dq
from jax import jit

dq.set_precision('double')
dq.set_progress_meter(False)

rtol_atol = 1e-8
method = dq.method.Tsit5(rtol=rtol_atol, atol=rtol_atol)
options = dq.Options(save_propagators=True, progress_meter=False, t0=0)

##########
# Time-evolution to find the propagator
##########
def get_Hamiltonian(H0, H1, A, omega_d):
    """Return the time-dependent Hamiltonian."""
    return dq.constant(H0) + dq.modulated(lambda t: A * jnp.cos(omega_d * t), H1)

def propagator(H, omega_d):
    """Get the propagator given a time-dependent Hamiltonian and frequency."""
    T = 2.0 * jnp.pi / omega_d
    ts = jnp.array([T])
    seprop_result = dq.sepropagator(H, ts, method=method, options=options)
    return seprop_result.final_propagator

##########
# Diagonalize propagator to get Floquet solution
##########
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

##########
# Package everything together
##########
@jit
def dq_basic(H0, H1, A, omega_d, **_):
    H = get_Hamiltonian(H0, H1, A, omega_d)
    U = propagator(H, omega_d)
    out = floquet_dq_basic(U)
    return out

@jit
def cayley(H0, H1, A, omega_d, cayley_phi=0, **_):
    H = get_Hamiltonian(H0, H1, A, omega_d)
    U = propagator(H, omega_d)
    out = floquet_cayley(U, cayley_phi)
    return out

SOLVERS = {
    'dq_basic': dq_basic,
    'cayley': cayley
}