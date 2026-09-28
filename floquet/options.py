from __future__ import annotations

from .utils.file_io import Serializable
import dynamiqs as dq


class Options(Serializable):
    def __init__(
        self,
        fit_range_fraction: float = 1.0,
        floquet_sampling_time_fraction: float = 0.0,
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,
        max_steps: int = 30_000,
        num_cpus: int = 1,
        save_floquet_modes: bool = False,
        atol = 1e-8,
        rtol = 1e-8,
    ):
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
        
