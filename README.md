# Compare various solvers for floquet. 

## Install

1. Install `uv` locally. Instructions [https://docs.astral.sh/uv/getting-started/installation/](here). 
2. Clone this repo. Inside the directory run `uv init --name floquet_gpu` to install the necessary dependencies. 
3. The benchmarking shell scripts are straightforward to run (e.g. `./submit.sh`). This submits all the benchmarking scripts as a series of SLURM array jobs.
You might have to delete a whole bunch of output files.
4. To run a notebook, select the virtual environment `floquet_gpu`.

## How it works:
1. The script solves the Floquet problem for a range of Hilbert space dimensions, with multiple trial runs per dimension. Here, each trial run defines a pair of random matrices, $H_0$ and $H_1$, and solves the Floquet problem for the Hamiltonian $H(t) = H_0 + A \cos (\omega_d t) H_1$. $H_0$ and $H_1$ are defined to be Hermitian matrices, with unit spectral norm. Importantly, since JAX uses pseudo-random generations, the same matrices can be generated across various solvers. This allows us to validate the computed Floquet modes and quasienergies, with a reliable solver (named "basic" here), such as qutip. 

2. Running `./submit.sh` first runs a set of "basic" jobs. This should be the code that you can rely on the most; i.e. your source of truth. Currently this is set up to solve the Floquet problem using QuTiP, on the CPU. 

3. Then, it runs the `basic_dq` and `cayley` solver on all the specified devices, with a nested loop. Each job runs the solver **three** times: a warm-up round for compilation, then a *timed* round with no profiler attached, then a *traced* round that captures the xplane. The timed and traced rounds are deliberately separate: starting and stopping the profiler costs ~100 ms on GPU and ~150 ms on CPU, which at small `d` is orders of magnitude more than the solve itself. `t_total` is the untraced number and is the one to plot; `t_total_traced` is kept only so profiler overhead stays visible.

4. Each device is swept over one or more **run variants**, configured by `GPU_VARIANTS` / `CPU_VARIANTS` at the top of `submit.sh`. Each entry is a `"tag|VAR=value;VAR=value"` pair whose assignments are exported into the benchmark process, so a variant can set `XLA_FLAGS`, library environment variables (cuBLAS emulation, etc.), or any combination. Every (device × variant × solver × dim × run) combination is submitted. Results land in `out/{device}_{tag}/`, and every row records `tag` plus a snapshot of the `XLA_*`/`CUBLAS_*`/`JAX_*`/`CUDA_*` environment actually in effect — so variants can never be mixed in one file, and the provenance cannot drift from what really ran.

   **Device time is not comparable across variants.** With `+WHILE` command buffers the Tsit5 loop runs as a single CUDA graph, and the profiler reports graph *launch* cost instead of kernel cost — `device_busy_ns` drops ~7× at d=4096 while the actual runtime is unchanged. Compare `t_total` across variants; compare `device_busy_ns` only within one. The propagator breakdown must be built from a `nocmdbuf` trace.

5. Once all the jobs for one (device, variant) are complete, `consolidate.py` merges them into `out/{device}_{tag}.npy`. Note that sometimes jobs may fail — if they exceed the time-limit (currently set to 1 hour), the allotted CPU memory (currently 10GB) or the GPU memory. `consolidate.py` ignores any failed job and reports how many were found per solver.

   Consolidation **merges** on `(dim, run_index)` rather than replacing: a later batch of extra runs tops the file up instead of overwriting it, re-running an existing `run_index` replaces just that row, and a solver whose raw files are absent keeps whatever is already consolidated. That is what makes step 9 safe.

6. The actual traces can be found in `out/{device}_{tag}/traces/{solver}_d{d}_run{run_index}/`. To browse one interactively, point `xprof` at the traces directory: `uv run xprof --logdir out/{device}_{tag}/traces`. Then open the printed `http://localhost:6006/` URL; each `{solver}_d{d}_run{run_index}` shows up as a separate run in the dropdown. Each trace directory also contains a `memory.prof` pprof snapshot of live device buffers (`jax.profiler.save_device_memory_profile`), viewable with e.g. `go tool pprof -top -unit=MB memory.prof`.

7. For the per-kernel breakdown of the integration phase, run `uv run python profiling.py --trace-root out/{device}_nocmdbuf/traces`. It writes `propagator_breakdown.npy` beside the traces. It raises rather than returning a partial result if its assumptions break (no stream events found, no post-processing boundary, or too many uncategorised kernels), and cross-checks its kernel totals against `xprof`.

8. **Reclaiming space: the raw output directories are disposable.** A finished sweep leaves ~10,000 files and ~4.3 GB under `out/{device}_{tag}/` (per-run `.npy`, SLURM logs, and one trace directory per run). None of it is needed once two things are true:

   - every row carries a `kernels` summary, so no analysis re-reads a trace. New runs get this automatically from `benchmark.py`; for older runs use `uv run python backfill_kernels.py`, which updates `out/{name}.npy` in place (it only *adds* a field, so it cannot lose data);
   - consolidation merges rather than replaces (step 5), so deleting the raw files cannot cost you rows later.

   Then `out/*.npy` plus `plot.ipynb` — about 13 MB and 14 files — is the complete, self-contained result set, and `rm -rf out/*/` is safe.

   Keep exactly **one** file per (device, variant), the one `consolidate.py` writes. An earlier version of this workflow kept a `_k.npy` sidecar with the kernel summaries and had the notebook prefer it; when a sweep was later extended, consolidation updated `out/{name}.npy` while the notebook went on reading the stale sidecar, and the new dimensions simply never appeared. The notebook now reads `out/{name}.npy` only, and prints the row count and dimension range of everything it loads — if that does not match the sweep you just ran, re-run `consolidate.py`.

   What you give up: `profiling.py`'s per-category propagator breakdown (`phase_breakdown`/`build`) still parses raw traces, as does `backfill_kernels.py`. If you may want either again, keep the traces for one faithful-attribution variant (`*_all_cmdbuf_off`) and delete the rest. The SLURM `*_err_*.txt` logs also go — they are how job failures get diagnosed, so check them before deleting.

9. **Validating the `ozaki` variant.** `CUBLAS_EMULATE_DOUBLE_PRECISION=1` asks cuBLAS to emulate FP64 GEMM with lower-precision tensor-core products (Ozaki-I) instead of native FP64 units. Three things must hold before its timings mean anything, and none of them is checked automatically:

   - **Is it active?** If the installed cuBLAS predates the feature, the variables are ignored silently and `ozaki` is a relabelled duplicate of `nocmdbuf`. Check the cuBLAS version (`python -c "import jax; print(jax.print_environment_info())"` and the CUDA toolkit version), and confirm the GEMM kernel names in the `ozaki` trace differ from the `nocmdbuf` trace — if the kernel names are identical, nothing changed.
   - **Does it apply to ZGEMM?** This workload is complex128 end to end. The emulation is documented primarily for real `DGEMM`; if complex is not covered, the variant is a no-op here and the kernel names will again be unchanged.
   - **Is it still accurate?** Emulated FP64 is not bit-identical to native FP64, and Tsit5 is *adaptive* at `rtol = atol = 1e-8`. Reduced GEMM accuracy changes the number of accepted steps, so a wall-clock difference between `ozaki` and `nocmdbuf` conflates "faster matmul" with "different amount of work". Compare the returned quasienergies against the `nocmdbuf` run at the same seed before reading anything into the timings — the seeds are reproducible across variants by construction (`jrand.fold_in(jrand.key(run_index), d)`).

## Some choices I made:

1. Each trial is run as a seperate job. I was worried that if multiple trials are run within the same job, they may occupy memory and slow down future trials. 

2. Avoiding any post-processing or comparison to the ground truth. Just focusing on the two promising solvers: `basic_dq` and `cayley`. 

3. Temporarily deleted `bench_solvers.py`. I should bring this back at some point. 

4. Memory is measured starting from before `H_0` and `H_1` are created. Both high-water marks (host RSS and, on GPU, `peak_bytes_in_use`) are read immediately after the timed run and *before* the traced run — `ru_maxrss` is a high-water mark, so parsing the xplane protobuf afterwards would otherwise be charged to the solver.

   On GPU, note that `mem_total` (host RSS) is dominated by a ~3.8 GB constant from CUDA library initialisation and barely moves with `d`; `mem_gpu` is the meaningful signal there. On CPU, `mem_total` tracks `d` properly.

5. For all the JAX-based solvers (regardless of JIT), the solver is "warmed-up" with a single run. This triggers any compilation defined in my code, as well as any hidden compilation, defined by `dynamiqs`. I'm not sure if this is the best decision, but you should keep this in mind while interpreting the results.