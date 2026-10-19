from __future__ import annotations

from .utils.file_io import Serializable
import dynamiqs as dq
from jax import default_backend
import warnings

class Options(Serializable):
    def __init__(
        self,
        device: str | None = None,
        fit_range_fraction: float = 1.0,
        floquet_sampling_time_fraction: float = 0.0,
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,
        max_steps: int = 30_000,
        num_cpus: int = 1,
        save_floquet_modes: bool = False,
        atol = 1e-8,
        rtol = 1e-8,
        amp_chunk: int = 4,
        fit_freq_chunk: int | None = None,
        default_eigh: str | None = None,
        retry_eigh: str | None = "JACOBI",
    ):

        self.device = device
        if device is None:
            self.device = default_backend()
        if self.device != default_backend() and self.device != 'cpu':
            warnings.warn(
                f"The requested device ({device}) is not available. Falling back to default available: {default_backend()}.",
                stacklevel=3,
            )
            self.device = default_backend()
        dq.set_device(self.device)
        dq.set_precision("double")
        
        if fit_range_fraction <= 0 or fit_range_fraction > 1:
            raise ValueError(
                f"Must have 0 < fit_range_fraction <= 1 but got {fit_range_fraction}"
            )
        self.fit_range_fraction = fit_range_fraction
        self.floquet_sampling_time_fraction = floquet_sampling_time_fraction
        if fit_cutoff < 1:
            raise ValueError(f"Must have fit_cutoff > 0 but got {fit_cutoff}")
        self.fit_cutoff = fit_cutoff
        if overlap_cutoff > 1 or overlap_cutoff < 0.7:
            raise ValueError(
                f"Must have 0.7 <= overlap_cutoff <= 1 but got {overlap_cutoff}"
            )
        self.overlap_cutoff = overlap_cutoff
        self.num_cpus = num_cpus
        self.save_floquet_modes = save_floquet_modes
        
        self.max_steps = max_steps
        self.atol = atol
        self.rtol = rtol

        self.dq_method = dq.method.Tsit5(atol=atol, rtol=rtol, max_steps=max_steps)
        self.dq_options = dq.Options(save_propagators=True, progress_meter=False, t0=0)
        self.cayley_phi = 0

        self.amp_chunk = amp_chunk

        # Number of drive frequencies per chunk for the online displaced state fit.
        # None fits all points at once.
        if fit_freq_chunk is not None and fit_freq_chunk < 1:
            raise ValueError(f"Must have fit_freq_chunk > 0 but got {fit_freq_chunk}")
        self.fit_freq_chunk = fit_freq_chunk


        # On GPU, JAX's default Hermitian eigensolver for matrices larger than 32 x 32
        # occasionally fails to converge. If it fails, then JAX then silently returns NaN. 
        # This wrapper recomputes the failed points. with the retry_eigh solver. Jacobi is
        # way slower than the default at dimension, so it is only used for these few points.
        # Jacobi is GPU-only; on CPU the retry falls back to the default.
        # The options below store the default and retry jax.lax.linalg.EighImplementation method. 
        # Either "QR", "JACOBI", or None.
        if default_eigh in ["QR", "JACOBI", None]:
            self.default_eigh = default_eigh
        else:
            warnings.warn(
                f"Unrecognized default_eigh = {default_eigh}. Falling back to the JAX default.",
                stacklevel=3,
            )
            self.default_eigh = None

        if retry_eigh in ["QR", "JACOBI", None]:
            self.retry_eigh = retry_eigh
        else:
            warnings.warn(
                f"Unrecognized retry_eigh = {retry_eigh}. Falling back to the JAX default.",
                stacklevel=3,
            )
            self.retry_eigh = None

        # Jacobi eigh is not implemented on CPU
        if self.default_eigh == "JACOBI" and self.device == 'cpu':
            self.default_eigh = None
        if self.retry_eigh == "JACOBI" and self.device == 'cpu':
            self.retry_eigh = None
