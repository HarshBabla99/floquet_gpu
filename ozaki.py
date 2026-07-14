"""Ozaki-scheme accurate GEMM emulation via INT8 tensor-core-mappable matmuls,
and a mixed-precision iterative-refinement solve built on top of it.

Reference: Ozaki, Ogita, Oishi, Rump (2012); Ozaki, Uchino, Imamura,
"Ozaki Scheme II" (arXiv:2504.08009).
"""
import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

INT8_MAX = 127

##
# Real-valued extraction + GEMM emulation
##
def _extract_rows(residual, n_slices):
    """Split a real float64 matrix into n_slices row-scaled int8 slices.

    Each row shares one power-of-two scale, so scale can be factored out of
    the K-contraction later.
    """
    N = residual.shape[0]
    slices, scales = [], []
    for _ in range(n_slices):
        row_max = jnp.max(jnp.abs(residual), axis=1)
        zero_mask = row_max == 0
        row_max_safe = jnp.where(zero_mask, 1.0, row_max)
        e0 = jnp.floor(jnp.log2(row_max_safe))
        scale = jnp.where(zero_mask, 1.0, jnp.exp2(e0 - 6.0))

        q = jnp.clip(jnp.round(residual / scale[:, None]), -INT8_MAX, INT8_MAX)
        v = q * scale[:, None]
        residual = residual - v

        slices.append(q.astype(jnp.int8))
        scales.append(scale)
    return jnp.stack(slices), jnp.stack(scales)


def _extract_cols(residual, n_slices):
    """Split a real float64 matrix into n_slices column-scaled int8 slices."""
    M = residual.shape[1]
    slices, scales = [], []
    for _ in range(n_slices):
        col_max = jnp.max(jnp.abs(residual), axis=0)
        zero_mask = col_max == 0
        col_max_safe = jnp.where(zero_mask, 1.0, col_max)
        e0 = jnp.floor(jnp.log2(col_max_safe))
        scale = jnp.where(zero_mask, 1.0, jnp.exp2(e0 - 6.0))

        q = jnp.clip(jnp.round(residual / scale[None, :]), -INT8_MAX, INT8_MAX)
        v = q * scale[None, :]
        residual = residual - v

        slices.append(q.astype(jnp.int8))
        scales.append(scale)
    return jnp.stack(slices), jnp.stack(scales)


def _check_accumulation_safety(K):
    # int32 accumulator over K terms of magnitude <= INT8_MAX**2 each.
    if K * INT8_MAX**2 >= 2**31:
        max_K = 2**31 // INT8_MAX**2
        raise ValueError(
            f'contraction dimension K={K} unsafe for int32 accumulation '
            f'(need K <= {max_K}).'
        )


@functools.partial(jax.jit, static_argnames=('n_slices',))
def _ozaki_matmul_real(A, B, n_slices):
    N, K = A.shape
    M = B.shape[1]
    A_slices, A_scales = _extract_rows(A, n_slices)
    B_slices, B_scales = _extract_cols(B, n_slices)

    dn = (((1,), (0,)), ((), ()))
    C = jnp.zeros((N, M), dtype=jnp.float64)
    for i in range(n_slices):
        inner = jnp.zeros((N, M), dtype=jnp.float64)
        # Triangular truncation: pairs with i+j > n_slices-1 fall below the
        # precision reachable with n_slices total, so they're skipped.
        for j in range(n_slices - i):
            P = jax.lax.dot_general(A_slices[i], B_slices[j], dn,
                                     preferred_element_type=jnp.int32)
            inner = inner + P.astype(jnp.float64) * B_scales[j][None, :]
        C = C + inner * A_scales[i][:, None]
    return C


def ozaki_matmul(A, B, n_slices=6):
    """Accurate A @ B via INT8-emulated GEMM(s). Supports real or complex128.

    n_slices trades accuracy for cost: each additional slice extracts ~7
    more mantissa bits and adds a triangular set of extra INT8 GEMMs.
    n_slices=8 is enough to fully reconstruct float64 (~53 bits); fewer
    slices give a cheaper, lower-accuracy result.
    """
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f'Incompatible shapes {A.shape} and {B.shape}.')
    _check_accumulation_safety(A.shape[1])

    if jnp.iscomplexobj(A) or jnp.iscomplexobj(B):
        Ar, Ai = jnp.real(A).astype(jnp.float64), jnp.imag(A).astype(jnp.float64)
        Br, Bi = jnp.real(B).astype(jnp.float64), jnp.imag(B).astype(jnp.float64)
        rr = _ozaki_matmul_real(Ar, Br, n_slices)
        ii = _ozaki_matmul_real(Ai, Bi, n_slices)
        ri = _ozaki_matmul_real(Ar, Bi, n_slices)
        ir = _ozaki_matmul_real(Ai, Br, n_slices)
        return (rr - ii) + 1j * (ri + ir)

    return _ozaki_matmul_real(A.astype(jnp.float64), B.astype(jnp.float64), n_slices)


##
# Mixed-precision iterative-refinement solve
##
def ozaki_solve(A, B, n_slices=6, max_iters=3):
    """Solve A @ X = B via GMRES-IR-style iterative refinement: a cheap
    single-precision LU factorization as preconditioner, with the residual
    at every refinement step computed by the accurate ozaki_matmul.
    """
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f'A must be square, got {A.shape}.')
    b_is_vec = B.ndim == 1
    if b_is_vec:
        B = B[:, None]
    if B.shape[0] != A.shape[0]:
        raise ValueError(f'Incompatible shapes A={A.shape}, B={B.shape}.')

    complex_in = jnp.iscomplexobj(A) or jnp.iscomplexobj(B)
    lp_dtype = jnp.complex64 if complex_in else jnp.float32
    hp_dtype = jnp.complex128 if complex_in else jnp.float64

    A_hp = A.astype(hp_dtype)
    B_hp = B.astype(hp_dtype)

    lu, piv = jsla.lu_factor(A_hp.astype(lp_dtype))
    x = jsla.lu_solve((lu, piv), B_hp.astype(lp_dtype)).astype(hp_dtype)

    for _ in range(max_iters):
        R = B_hp - ozaki_matmul(A_hp, x, n_slices)
        d = jsla.lu_solve((lu, piv), R.astype(lp_dtype)).astype(hp_dtype)
        x = x + d

    return x[:, 0] if b_is_vec else x
