import jax.numpy as jnp
from jax import devices as jdevices
import dynamiqs as dq
import numpy as np

from resource import getrusage, RUSAGE_SELF
from sys import platform

dq.set_precision('double')
dq.set_progress_meter(False)

####################################################################################################
# Linalg utils

def nonunitarity(U):
    """Compute deviation of the integrated propagator from unitarity.
    Expected to grow linearly in ||H||*T. 
    """
    U = U.to_jax() if hasattr(U, 'to_jax') else jnp.asarray(U)
    D = U.conj().T @ U - jnp.eye(U.shape[-1], dtype=U.dtype)
    return float(jnp.abs(D).max())

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

####################################################################################################
# Memory utils

def peak_rss_mb():
    """Peak memory of this process (on host), in MB."""
    ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux, bytes on macOS.
    return ru_maxrss / (1024 if platform != 'darwin' else 1024 * 1024)

def gpu_peak_mb(is_gpu):
    """Peak memory of this process (on the GPU), in MB. None if not supported"""
    if not is_gpu:
        return None
    stats = jdevices()[0].memory_stats()
    if not stats:
        return None
    return stats['peak_bytes_in_use'] / 1024**2

####################################################################################################
# File IO

def load_rows(path):
    """Read all rows sequentially from a .npy file written in append mode."""
    rows = []
    with open(path, 'rb') as f:
        while True:
            try:
                rows.append(np.load(f, allow_pickle=True).item())
            except (EOFError, ValueError):
                break
    return rows