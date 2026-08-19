import numpy as np

from jax.numpy.linalg import norm
import jax.random as jrand

import dynamiqs as dq
import scqubits as scq

####################################################################################################
# Random matrices

def random_hermitian(shape, key):
    """Generate a random Hermitian matrix with unit norm."""
    H = dq.random.herm(key, shape)
    _norm = norm(H.to_jax(), 2)
    return H / _norm

def random_hermitian_matrices(run_index, d):
    k0, k1, k2, k3 = jrand.split(jrand.fold_in(jrand.key(run_index), d), num=4)
    h0_scale=100
    H0 = h0_scale*random_hermitian((d, d), k0)
    H1 = random_hermitian((d, d), k1)
    A = float(jrand.uniform(k2, minval=0.5, maxval=1.5))
    omega_d = float(jrand.uniform(k3, minval=1.0, maxval=5.0))

    return H0, H1, A, omega_d

####################################################################################################
# Transmon + Vary-dim resonator

def get_transmon(omega_p=1, zeta=0.25, truncated_dim=20):
    """Generate a deterministic transmon. zeta=0.25 results in EJ/EC approx 100"""

    scq_qubit_params = {
        'EC' : omega_p * zeta/8,
        'EJ' : omega_p / zeta,
        'ncut': 41,
        'ng' : 0.0,
        'truncated_dim' : truncated_dim,
    }
    tmon = scq.Transmon(**scq_qubit_params)
    return tmon

def get_floquet_params_from_scq(hilbert_space, tmon, k2, k3):
    # Diagonalize
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0]
    evals = evals - evals[0]

    # Define the operators
    H0 = 2.0 * np.pi * dq.sparsedia_from_dict({0:evals})
    
    H1_np = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)
    drive_matelem = H1_np[0, 1]
    H1 = dq.asqarray(H1_np)

    # Drive params
    xi_sq = float(jrand.uniform(k2, minval=0.5, maxval=3.0))
    omega_d = float(jrand.uniform(k3, minval=1.2, maxval=7.0))
    A = 0.5 * np.sqrt(xi_sq) * (1 - omega_d**2)/(drive_matelem * omega_d)

    return H0, H1, A, omega_d

def transmon_vary_resonator_dim(run_index, d):
    k0, k1, k2, k3 = jrand.split(jrand.fold_in(jrand.key(run_index), d), num=4)

    # Deterministic transmon
    tmon = get_transmon()

    # random resonator and coupling
    # resonator freq = 1.2 to 3.0 times qubit frequency
    # g = (1/20) to (1/30) times qubit freq
    omega_r = float(jrand.uniform(k0, minval=1.2, maxval=3.0))
    g = float(jrand.uniform(k1, minval=(1/30), maxval=(1/20)))

    res = scq.Oscillator(E_osc = omega_r, truncated_dim=d)

    hilbert_space = scq.HilbertSpace([tmon, res])
    hilbert_space.add_interaction(
        g_strength = g,
        op1 = tmon.n_operator,
        op2 = res.annihilation_operator,
        add_hc = True
    )

    return get_floquet_params_from_scq(hilbert_space, tmon, k2, k3)