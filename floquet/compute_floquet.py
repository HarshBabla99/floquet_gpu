from __future__ import annotations

from functools import partial

import dynamiqs as dq

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.lax import map as lax_map
from jax.lax.linalg import eigh as lax_eigh, EighImplementation

from .model import Model
from .options import Options

@partial(jit, static_argnames=("options", "eigh_impl"))
def compute_floquet(
    model: Model,
    omega_d: float,
    amp: float,
    freq_idx: int = 0,
    *,
    options: Options,
    eigh_impl: str | None = None,
) -> tuple[Array, Array]:
    """Run one instance of the problem for a pair of drive frequency and amp.

    Parameters:
        model: Model (an equinox Module, so its arrays are traced by jit).
        omega_d: Drive frequency.
        amp: Drive amplitude.
        freq_idx: Index of omega_d in model.omega_d_values, for models whose H0
            depends on the drive frequency.
        options: Options. Static: hashed by jit, so changing it triggers a recompile.
        eigh_impl: jax.lax.linalg.EighImplementation method. Either "QR", "JACOBI", or None.
            None defaults to JAX's default (i.e. JACOBI on GPU for smaller matrices, QR on CPU and 
            larger matrices on GPU). 

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
    seprop_result = dq.sepropagator(model.hamiltonian(omega_d, amp, freq_idx), ts, 
                                    method=options.dq_method, 
                                    options=options.dq_options)

    # undo the interaction picture, then rotate back to the lab basis
    U_tilde = seprop_result.final_propagator.elmul(
        jnp.exp(-1j * model.H0_evals_at(freq_idx) * T)[:, None]
    )
    
    # stay in the H0 eigenbasis, where the bare states are the basis states
    # (to go to the original basis: propagator = evecs @ U_tilde @ evecs.dag())
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
    if eigh_impl is None:
        _, prop_evecs = jnp.linalg.eigh(H)
    else:
        # lax.linalg.eigh returns (evecs, evals)
        prop_evecs, _ = lax_eigh(H, implementation=EighImplementation[eigh_impl])

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

@partial(jit, static_argnames=("options",))
def _compute_floquet_grid(model: Model, *, options: Options) -> tuple[Array, Array]:
    """Floquet solver for every (omega_d, amp) in the model.
    
    Vectorized over all drive frequencies, but amplitudes are iterated over
    sequentially of chunk size amp_chunk.

    Returns:
        floquet_modes: shape (n_omega, n_amp, dim, dim)
        quasienergies: shape (n_omega, n_amp, dim)
    """
    omega_d = jnp.asarray(model.omega_d_values)
    freq_idxs = jnp.arange(len(omega_d))
    over_freq = vmap(
        lambda w, i, a: compute_floquet(
            model, w, a, i, options=options, eigh_impl=options.default_eigh
        )
    )
    modes, energies = lax_map(
        lambda a_row: over_freq(omega_d, freq_idxs, a_row),
        jnp.asarray(model.drive_amplitudes),
        batch_size=options.amp_chunk,
    )
    return jnp.swapaxes(modes, 0, 1), jnp.swapaxes(energies, 0, 1)


def compute_floquet_grid(model: Model, *, options: Options) -> tuple[Array, Array]:
    """Floquet modes and quasienergies for every (omega_d, amp) in the model.
    Tries with two different eigensolvers, because very rarely JAX's default 
    eigensolver fails silently, i.e. returns NaNs without warning.

    Returns:
        floquet_modes: shape (n_omega, n_amp, dim, dim)
        quasienergies: shape (n_omega, n_amp, dim)

    Raises:
        FloatingPointError: if any point is still non-finite after the retry.
    """
    modes, energies = _compute_floquet_grid(model, options=options)
    ok = jnp.all(jnp.isfinite(modes), axis=(-2, -1)) & jnp.all(jnp.isfinite(energies), axis=-1)
    bad_idxs = jnp.argwhere(~ok) # (n_bad, 2) indices into (omega, amp)
    if len(bad_idxs) == 0:
        return modes, energies

    omega_d = jnp.asarray(model.omega_d_values)
    amps = jnp.asarray(model.drive_amplitudes)

    # recomputing sequentially here, because the no. of bad points varies. 
    # we don't want to re-JIT every time the shape changes. 
    # usually we only have a few bad points, so the loop should be fast. 
    fixed = [
        compute_floquet(
            model, omega_d[i], amps[j, i], i, options=options, eigh_impl=options.retry_eigh
        )
        for i, j in bad_idxs
    ]

    # replace the bad data with the recomputed values.
    rows, cols = jnp.asarray(bad_idxs[:, 0]), jnp.asarray(bad_idxs[:, 1])
    modes = modes.at[rows, cols].set(jnp.stack([m for m, _ in fixed]))
    energies = energies.at[rows, cols].set(jnp.stack([e for _, e in fixed]))

    # see if any of the recomputed points are still bad
    ok = jnp.all(jnp.isfinite(modes), axis=(-2, -1)) & jnp.all(jnp.isfinite(energies), axis=-1)
    still_bad = jnp.argwhere(~ok)
    
    if len(still_bad) > 0:
        raise FloatingPointError(
            "Floquet modes still non-finite after retry at (omega_idx, amp_idx) "
            f"{still_bad.tolist()}"
        )
    
    print(f"recomputed {len(bad_idxs)} non-finite point(s) with eigh_impl={options.retry_eigh}", 
          flush=True)
    
    return modes, energies


# reflect the number of JIT-compiled functions in the cache (for debugging)
compute_floquet_grid._cache_size = _compute_floquet_grid._cache_size
