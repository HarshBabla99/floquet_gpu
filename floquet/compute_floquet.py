from __future__ import annotations

from functools import partial

import dynamiqs as dq
import jax.numpy as jnp
from jax import Array, jit

from .model import Model
from .options import Options

@partial(jit, static_argnames=("options",))
def compute_floquet(
    model: Model, omega_d: float, amp: float, *, options: Options
) -> tuple[Array, Array]:
    """Run one instance of the problem for a pair of drive frequency and amp.

    Parameters:
        model: Model (an equinox Module, so its arrays are traced by jit).
        omega_d: Drive frequency.
        amp: Drive amplitude.
        options: Options. Static: hashed by jit, so changing it triggers a recompile.

    Returns:
        Floquet modes. Shape: `(hilbert_dim, hilbert_dim)`, where the first index
            labels the mode and the second the basis vector components.
        Quasienergies. Shape: `(hilbert_dim,)`.

    """

    #### 
    # integrate to find the propagator
    #### 
    # time for a single period
    T = 2.0 * jnp.pi / omega_d
    ts = jnp.array([T])

    # interaction-picture propagator W(T), in the H0 eigenbasis
    seprop_result = dq.sepropagator(model.hamiltonian(omega_d, amp), ts, 
                                    method=options.dq_method, 
                                    options=options.dq_options)

    # undo the interaction picture, then rotate back to the lab basis
    U_tilde = seprop_result.final_propagator.elmul(jnp.exp(-1j * model.H0_evals * T)[:, None])
    
    # if H0 isn't diag: propagator = evecs @ U_tilde @ evecs.dag()
    U = U_tilde

    #### 
    # diagonalize the propagator via Cayley
    #### 
    I = dq.eye_like(U)

    # turn both into jax.numpy arrays
    I = I.to_jax()
    U = U.to_jax()

    # Issue when U has an evalue of -1; causes a singularity in (I+U)^{-1}. 
    # Solution: rotate with a random phase. Eigenvectors are unchanged.
    W = jnp.exp(1j * options.cayley_phi) * U

    # construct the Hermitian matrix
    H = 1j * jnp.linalg.solve(I + W, I - W)
    H = 0.5 * (H + H.conj().T)

    # diagonalize hermitian
    _, prop_evecs = jnp.linalg.eigh(H)

    # Recover evals of U: diag(V^dag U V) = (V^dag U V)_ii = \sum_j (V^*)_{ji} (U V)_{ji}
    # where V = prop_evecs
    prop_evals = jnp.sum(jnp.conj(prop_evecs) * (U @ prop_evecs), axis=0)

    # quasienergies, folded into the first Brillouin zone (-pi/T, pi/T]
    f_energies = -jnp.angle(prop_evals) / T
    f_energies = jnp.mod(f_energies + 0.5 * omega_d, omega_d) - 0.5 * omega_d

    # remove the global phase on the maximum-magnitude component of each mode
    pivot = jnp.argmax(jnp.abs(prop_evecs), axis=0)
    pv = jnp.take_along_axis(prop_evecs, pivot[None, :], axis=0)[0]
    phase = pv / jnp.abs(pv)
    prop_evecs = prop_evecs * jnp.conj(phase)[None, :]

    # sort by quasienergy
    perm = jnp.argsort(f_energies)

    # intermediate time evolve
    # TODO: SKIPPED FOR NOW
    f_modes_t = prop_evecs

    return f_modes_t.T[perm], f_energies[perm]