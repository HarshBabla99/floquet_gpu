"""Analysis of the device traces captured by `benchmark.py`.

Two levels of granularity, both reading the same `.xplane.pb` files:

1. `summarize_trace()` - per-*HLO op* self time for a single trace. Called by
   `benchmark.py` right after each capture, so the numbers land in the
   benchmark `.npy` rows.
2. `phase_breakdown()` / `build()` - per-*CUDA kernel* categories for the
   propagator (ODE-integration) phase only. `op_self_ns` above is keyed by
   HLO op (e.g. `command_buffer_32`), which lumps the whole Tsit5 loop body
   together; to see what the integration actually spends time on (dense matmul
   vs. adaptive-stepper overhead) we go one level deeper, into the kernel
   events.

Splitting propagator from diagonalisation
-----------------------------------------
`solvers.py` marks the two stages with `jax.named_scope`, which tags every XLA
instruction with the enclosing scope. `phase_map()` reads that tagging back out
of the compiled HLO, giving the compiler's own instruction-to-phase assignment;
`summarize_trace(pmap=...)` and `kernel_summary(pmap=...)` then join it against
the trace on `hlo_op` and report `phase_self_ns` / `kernels['phases']`.

This replaces the older approach of cutting the timeline at the first
post-processing op. That heuristic is still here as `phase_breakdown()` for
traces captured before the markers existed, but it is genuinely fragile: `POST`
has to name an op that occurs only after the ODE, and when cuBLAS matmuls began
appearing as bare `custom-call`s it cut at the first matmul *inside* the
integration and reported a 4 ms propagator against a 1700 ms remainder. The
scope-based map has no such failure mode - ops like `dot_general`, which really
do occur in both phases, are assigned per instruction rather than per name.

Both are best-effort reconstructions of what the profiler recorded. They now
fail loudly rather than returning an empty or half-populated result, because a
silently-empty breakdown is indistinguishable from a real one once it reaches a
figure. See `phase_breakdown` for the specific assumptions that are checked.

CAUTION: device-time attribution is not comparable across XLA configurations. If
a while-loop is captured into a CUDA graph
(`--xla_gpu_enable_command_buffer=+WHILE`), the profiler reports graph *launch*
cost rather than kernel cost and `device_busy_ns` collapses by ~7x while actual
runtime is unchanged. Every row records the `tag` and raw `xla_flags` it ran
under; group by those before comparing.

Run as a script to write {dim: {category: mean_ms}} to
<trace-root>/../propagator_breakdown.npy:

    uv run python profiling.py --trace-root out/gpu_h200_nocmdbuf/traces
"""
import argparse
import glob
import os
import re
import warnings
from collections import defaultdict

import jax
import numpy as np


def _find_xplane(trace_dir):
    """Locate the .xplane.pb file jax.profiler.trace() writes under trace_dir."""
    matches = glob.glob(os.path.join(trace_dir, 'plugins', 'profile', '*', '*.xplane.pb'))
    if not matches:
        return None
    # trace() writes exactly one xspace file per call; if more exist, take the newest.
    return max(matches, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Per-HLO-op summary of a single trace
# ---------------------------------------------------------------------------

def _self_times_ns(events):
    """Compute exclusive ("self") duration per op for one timeline.

    On CPU, a single host thread executes hlo_ops as genuine nested function
    calls: a `while` op's event spans its whole loop body, and the body's
    ops - `dot_general` etc. - are nested inside it as their own hlo_op-tagged
    events on the same line. Summing raw durations there double-counts
    nested time, so each op is credited only with the time not already
    claimed by an event it strictly contains (start <= child.start and
    child.end <= end), same as flame-graph "self time".

    On GPU, a device stream line multiplexes several physical engines
    (e.g. "Stream #13(Memset,MemcpyD2H,Compute,MemcpyD2D)"), so events can
    genuinely overlap in wall-clock time without one containing the other
    (concurrent copy/compute, or timestamp jitter between back-to-back
    kernels). That is real concurrent work, not double-counted nesting, so
    only *strict containment* is treated as nesting; partial/crossing
    overlaps are left alone and each event keeps its full duration.

    Two known limits of that rule, both specific to genuinely-concurrent GPU
    events on one line (neither has been observed in the traces collected so
    far, and `summarize_trace` warns if the first one occurs):
      - if two *siblings* overlap each other inside a common parent, both are
        subtracted from the parent and it can end up with negative self time;
      - two concurrent events with byte-identical start/end satisfy `<=`
        containment, so one is treated as nested in the other and contributes
        no self time.

    `events` is an iterable of (start_ns, end_ns, duration_ns, op_name).
    Returns {op_name: self_ns}.
    """
    events = sorted(events, key=lambda ev: (ev[0], -ev[1]))

    self_ns = {}
    stack = []  # entries: [start, end, duration, op_name, child_time_sum]

    def contains(outer, inner_start, inner_end):
        return outer[0] <= inner_start and inner_end <= outer[1]

    def close(entry):
        _, _, duration, op, child_sum = entry
        self_ns[op] = self_ns.get(op, 0.0) + (duration - child_sum)

    for start, end, duration, op in events:
        # Pop any open frame that does not strictly contain this event -
        # either it already ended, or this event only partially overlaps it.
        while stack and not contains(stack[-1], start, end):
            closed = stack.pop()
            close(closed)
            if stack:
                stack[-1][4] += closed[2]
        stack.append([start, end, duration, op, 0.0])

    while stack:
        closed = stack.pop()
        close(closed)
        if stack:
            stack[-1][4] += closed[2]

    return self_ns


def summarize_trace(trace_dir, top_k=15, pmap=None):
    """Aggregate per-HLO-op device time from a captured trace.

    Events tagged with an 'hlo_op' stat are compiled-XLA execution on a device
    (this holds regardless of backend/plane naming); everything else is host-side
    (Python dispatch, tracing, etc.) and is left to the interactive trace viewer.
    Within a plane/line, hlo_op events can nest (see `_self_times_ns`), so
    per-op time is the *self* time, not raw event duration - otherwise e.g. a
    `while` loop's time would be counted once for the loop and again for every
    op inside its body.

    Returns a dict with:
      op_self_ns:     {op_name: self_ns} over the whole trace. "Self" time is
                      exclusive of nested ops, so these sum without double
                      counting.
      device_busy_ns: total time the device spent executing compiled XLA ops,
                      i.e. sum(op_self_ns.values()). Compare against the row's
                      `t_total` to see how much of the runtime was device work
                      at all - the remainder is host-side dispatch and, at small
                      d, dominates.
      top_ops:        top_k (op_name, ns) pairs, sorted descending
      xplane_path:    path to the underlying .xplane.pb, or None if not found
    """
    xplane_path = _find_xplane(trace_dir)
    if xplane_path is None:
        return dict(op_self_ns={}, device_busy_ns=0, phase_self_ns={}, top_ops=[],
                    xplane_path=None)

    profile = jax.profiler.ProfileData.from_file(xplane_path)

    op_self_ns = {}
    for plane in profile.planes:
        for line in plane.lines:
            events = []
            for event in line.events:
                stats = dict(event.stats)
                op_name = stats.get('hlo_op')
                if op_name is None:
                    continue
                events.append((event.start_ns, event.end_ns, event.duration_ns, op_name))
            if not events:
                continue
            for op_name, self_ns in _self_times_ns(events).items():
                op_self_ns[op_name] = op_self_ns.get(op_name, 0.0) + self_ns

    # A negative self time means overlapping sibling events were both subtracted
    # from a common parent (see `_self_times_ns`). Warn rather than raise: this
    # corrupts one op's share, not the whole trace, and killing a 30-minute
    # benchmark job over it would cost more than it saves.
    negative = {op: ns for op, ns in op_self_ns.items() if ns < 0}
    if negative:
        warnings.warn(
            f'{len(negative)} op(s) got negative self time in {xplane_path} '
            f'(overlapping concurrent events on one line): {sorted(negative)[:5]}',
            RuntimeWarning, stacklevel=2)

    device_busy_ns = sum(op_self_ns.values())
    top_ops = sorted(op_self_ns.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    # Propagator / diagonalisation split, keyed on the compiler's own scope
    # metadata. This works on both backends because it needs only hlo_op names,
    # unlike the kernel-level split which requires GPU stream lines.
    phase_self_ns = {}
    if pmap:
        for op, ns in op_self_ns.items():
            phase_self_ns[phase_of(op, pmap) or 'unassigned'] = \
                phase_self_ns.get(phase_of(op, pmap) or 'unassigned', 0.0) + ns

    return dict(
        op_self_ns=op_self_ns,
        device_busy_ns=device_busy_ns,
        phase_self_ns=phase_self_ns,
        top_ops=top_ops,
        xplane_path=xplane_path,
    )


# ---------------------------------------------------------------------------
# Per-kernel breakdown of the propagator phase
# ---------------------------------------------------------------------------

# HLO ops that mark the END of the propagator phase.
#   cayley   -> jnp.linalg.solve (lu, triangular-solve) then eigh
#   dq_basic -> eig
# 'eig' is a prefix of 'eigh', so it covers both solvers.
#
# 'custom-call' MUST NOT be listed here. XLA:GPU lowers cuBLAS matmuls to
# custom-calls, so with cuBLAS graph-capture disabled
# (--xla_gpu_enable_command_buffer=FUSION) the Tsit5 loop's own GEMMs appear as
# bare `custom-call.N` ops. Including it snapped the boundary to the first matmul
# *inside* the integration and reported a ~1 ms propagator against a ~1700 ms
# "post-processing" phase. These four ops are unambiguous: Tsit5 is an explicit
# Runge-Kutta method and performs no linear solves or eigendecompositions.
POST = ('lu', 'eig', 'triangular-solve')

# The integration is matmul-dominated at every dimension we benchmark, so a
# propagator phase containing no GEMM at all means the boundary is misplaced.
MIN_GEMM_FRACTION = 0.05

# Fraction of propagator kernel time allowed to land in the 'other' bucket before
# we treat `categorize`'s rules as stale.
MAX_OTHER_FRACTION = 0.25


def categorize(name):
    """Map a CUDA kernel display-name to a propagator-phase category."""
    n = name.lower()
    if 'gemm' in n:
        return 'GEMM (complex matmul, O(d³))'
    if (n.startswith('loop_add') or n.startswith('loop_multiply')
            or 'wrapped_add' in n or 'wrapped_subtract' in n or 'wrapped_multiply' in n):
        return 'Tsit5 stage combine'
    if ('dynamic_update_slice' in n or 'loop_pad' in n or 'gather' in n
            or 'dynamic_slice' in n or 'concatenate' in n):
        return 'buffer/slice bookkeeping'
    if ('select' in n or 'reduce' in n or 'compare' in n
            or n.startswith('loop_and') or n.startswith('loop_or')):
        return 'adaptive step control'
    return 'other'


def _stream_events(prof, xplane_path):
    """Collect (start_ns, duration_ms, kernel_name, hlo_op) from GPU stream lines.

    Discovers device planes and hardware-stream lines by pattern instead of
    hardcoding '/device:GPU:0' and 'Stream #13', which are assigned by the
    profiler and shift with driver/JAX version or GPU count. Only lines named
    'Stream *' are read: those carry the real kernel events. Derived lines that
    some profiler versions add to the same plane ('XLA Ops', 'XLA Modules') are
    re-projections of the same time and would double-count.

    Raises RuntimeError, listing what the trace actually contains, if nothing
    matched - previously this returned {} and the dimension vanished from the
    figure without comment.
    """
    events = []
    seen = []
    for plane in prof.planes:
        if not plane.name.startswith('/device:'):
            continue
        for line in plane.lines:
            seen.append(f'{plane.name} | {line.name}')
            if not line.name.startswith('Stream'):
                continue
            for e in line.events:
                op = dict(e.stats).get('hlo_op')
                if op is None:
                    continue
                events.append((e.start_ns, e.duration_ns / 1e6, e.name, op))

    if not events:
        raise RuntimeError(
            f'No GPU stream kernel events found in {xplane_path}.\n'
            f'Device plane/line names present: {seen or "(no /device: plane at all)"}\n'
            'This is expected for a CPU-backend trace; phase_breakdown is GPU-only.')
    events.sort(key=lambda ev: ev[0])
    return events


def _assert_boundary_sane(events, boundary, xplane_path):
    """Check that the propagator/post-processing split landed where we think.

    The boundary is the first post-processing op (Cayley solve / eigh). Two ways
    it can be wrong, both of which used to pass silently:
      - no post-processing op found at all, so everything counts as propagator;
      - a POST-matching op fires at the very start, so the propagator phase
        collapses to nothing and the breakdown looks empty-but-valid.
    """
    if boundary == float('inf'):
        ops = sorted({op for _, _, _, op in events})
        raise RuntimeError(
            f'No post-processing op {POST} found in {xplane_path}; cannot tell '
            f'where integration ends. Ops present: {ops[:30]}')

    before = sum(d for s, d, _, _ in events if s < boundary)
    after = sum(d for s, d, _, _ in events if s >= boundary)
    if before <= 0:
        culprit = min((ev for ev in events if ev[0] == boundary), key=lambda ev: ev[0])
        raise RuntimeError(
            f'Propagator phase is empty in {xplane_path}: the first post-processing '
            f'op ({culprit[3]!r}, kernel {culprit[2]!r}) is also the first kernel in '
            f'the trace. Either POST matched something inside the Tsit5 loop (a '
            f'cuBLAS `custom-call` matmul is the likely culprit) or this trace has '
            f'no integration phase.')
    return before, after


def phase_breakdown(xplane_path, strict=True):
    """Return (categories, diagnostics) for the propagator phase of one trace.

    categories:  {category_name: ms} for kernels before the post-processing boundary
    diagnostics: propagator_ms / post_ms / other_frac / n_events, so callers can
                 see whether the split and the categorisation are trustworthy
                 instead of inferring it from a plausible-looking bar chart.
    """
    prof = jax.profiler.ProfileData.from_file(xplane_path)
    events = _stream_events(prof, xplane_path)

    post_starts = [s for s, d, n, op in events if any(op.startswith(p) for p in POST)]
    boundary = min(post_starts) if post_starts else float('inf')
    propagator_ms, post_ms = _assert_boundary_sane(events, boundary, xplane_path)

    cat = defaultdict(float)
    for s, d, n, op in events:
        if s < boundary:
            cat[categorize(n)] += d

    # A propagator phase with no GEMM in it is not a propagator phase. This is a
    # stronger check than "is it non-empty": when the boundary snapped to the
    # first in-loop matmul, a handful of stage-combine kernels still preceded it,
    # so the phase looked populated while containing 0.2% of the real work.
    gemm_ms = sum(v for k, v in cat.items() if k.startswith('GEMM'))
    if propagator_ms and gemm_ms / propagator_ms < MIN_GEMM_FRACTION:
        first_post = min((ev for ev in events if ev[0] == boundary), default=None)
        raise RuntimeError(
            f'Propagator phase in {xplane_path} contains almost no GEMM '
            f'({gemm_ms:.2f} of {propagator_ms:.2f} ms) - the integration is '
            f'matmul-dominated, so the boundary is misplaced. It landed at op '
            f'{first_post[3]!r} (kernel {first_post[2]!r}). Post-processing phase '
            f'is {post_ms:.1f} ms. Check that POST={POST} does not match an op '
            f'emitted inside the Tsit5 loop.')

    other_frac = cat.get('other', 0.0) / propagator_ms if propagator_ms else 0.0
    if other_frac > MAX_OTHER_FRACTION:
        unknown = sorted({n for s, d, n, op in events
                          if s < boundary and categorize(n) == 'other'})
        msg = (f'{100 * other_frac:.0f}% of propagator kernel time is uncategorised in '
               f'{xplane_path} - `categorize` rules are probably stale. '
               f'Unmatched kernels: {unknown[:15]}')
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    diagnostics = dict(propagator_ms=propagator_ms, post_ms=post_ms,
                       other_frac=other_frac, n_events=len(events))
    return dict(cat), diagnostics


# ---------------------------------------------------------------------------
# Exact phase assignment, read from the compiler
# ---------------------------------------------------------------------------

# `%name = shape op(...), metadata={op_name="jit(f)/SCOPE/..."}`
_HLO_INSTR = re.compile(r'%([\w.\-]+)\s*=.*?metadata=\{op_name="([^"]*)"')


def phase_map(hlo_text, markers=('PHASE_PROPAGATOR', 'PHASE_DIAGONALIZATION')):
    """{hlo_instruction_name: phase} from a compiled module's own metadata.

    `solvers.py` wraps each stage in `jax.named_scope`, which tags every XLA
    instruction it produces with the enclosing scope. Those tags survive
    compilation, so this is the compiler's own account of which phase each
    instruction belongs to - not an inference from op names or timestamps.

    The profiler does not carry the scope (an xplane event exposes only
    `hlo_op`, the bare instruction name), which is why the two have to be
    joined: build the map here, then look up each trace event's `hlo_op`.

    Get `hlo_text` from `jitted_fn.lower(*args).compile().as_text()`.
    """
    out = {}
    for name, op_name in _HLO_INSTR.findall(hlo_text):
        for mk in markers:
            if f'/{mk}/' in op_name or op_name.endswith(f'/{mk}'):
                out[name] = mk
                break
    return out


def phase_of(op_name, pmap):
    """Phase of one trace event's `hlo_op`, or None if the map does not cover it.

    XLA renames some instructions between the module we read and the one that
    runs (fusions in particular), so fall back to the un-suffixed stem before
    giving up.
    """
    if op_name in pmap:
        return pmap[op_name]
    stem = op_name.rsplit('.', 1)[0]
    return pmap.get(stem)


# ---------------------------------------------------------------------------
# Compact kernel-level summary (so traces need not be kept)
# ---------------------------------------------------------------------------

# Ops that are the propagator's matmuls once cuBLAS is not graph-captured.
_GEMM_PREFIXES = ('custom-call',)


def kernel_summary(trace_dir, top_kernels=25, pmap=None):
    """Condense a trace's GPU kernel timeline into a few hundred bytes.

    Every kernel-level number the analysis notebook needs, precomputed, so the
    (large) .xplane.pb files can be deleted after the run. Stored in each
    benchmark row as `kernels`.

    Returns {} for CPU traces, which have no GPU stream lines.
    """
    xplane_path = _find_xplane(trace_dir)
    if xplane_path is None:
        return {}
    prof = jax.profiler.ProfileData.from_file(xplane_path)
    try:
        events = _stream_events(prof, xplane_path)
    except RuntimeError:
        return {}                       # CPU backend: no GPU streams, nothing to summarise

    # include untagged events (memcpy etc.) when measuring occupancy, so "idle"
    # means the device really had nothing to do
    raw = []
    for plane in prof.planes:
        if not plane.name.startswith('/device:'):
            continue
        for line in plane.lines:
            if not line.name.startswith('Stream'):
                continue
            for e in line.events:
                raw.append((e.start_ns, e.duration_ns / 1e6, e.name,
                            dict(e.stats).get('hlo_op')))
    raw.sort()

    t0 = raw[0][0]
    span_ms = (raw[-1][0] + raw[-1][1] * 1e6 - t0) / 1e6
    busy_ms = sum(e[1] for e in raw)

    gaps, cur = [], raw[0][0] + raw[0][1] * 1e6
    for s, dur, _, _ in raw:
        if s > cur:
            gaps.append((s - cur) / 1e6)
        cur = max(cur, s + dur * 1e6)
    gaps = np.array(gaps) if gaps else np.zeros(1)
    big = gaps[gaps > 1.0]

    def agg(sel):
        sub = [e for e in raw if sel(e)]
        return dict(n=len(sub), ms=float(sum(e[1] for e in sub)))

    by_op = {}
    for e in raw:
        if e[3] is None:
            continue
        base = e[3].split('.')[0]
        slot = by_op.setdefault(base, [0, 0.0])
        slot[0] += 1
        slot[1] += e[1]
    by_op = {k: dict(n=v[0], ms=float(v[1])) for k, v in by_op.items()}

    by_kernel = {}
    for e in raw:
        key = e[2].split('<')[0][:48]
        slot = by_kernel.setdefault(key, [0, 0.0])
        slot[0] += 1
        slot[1] += e[1]
    by_kernel = dict(sorted(((k, dict(n=v[0], ms=float(v[1]))) for k, v in by_kernel.items()),
                            key=lambda kv: -kv[1]['ms'])[:top_kernels])

    # diagonalisation phase: kernels plus the wall span they occupy, which is
    # what exposes a launch-bound solver (many kernels, mostly idle)
    diag = [e for e in raw if e[3] and e[3].startswith(POST)]
    if diag:
        d0 = min(e[0] for e in diag)
        d1 = max(e[0] + e[1] * 1e6 for e in diag)
        dbusy = sum(e[1] for e in diag)
        diag_stats = dict(n=len(diag), busy_ms=float(dbusy),
                          idle_ms=float((d1 - d0) / 1e6 - dbusy),
                          start_ms=float((d0 - t0) / 1e6))
    else:
        diag_stats = dict(n=0, busy_ms=0.0, idle_ms=0.0, start_ms=float('nan'))

    def first_start(prefixes):
        for s, dur, n, op in raw:
            if op and op.startswith(prefixes):
                return float((s - t0) / 1e6)
        return float('nan')

    # Which CUDA kernels implement the matmul. Under FP64 emulation one logical
    # matmul expands into many differently-named kernels, so this is what shows
    # the emulation actually engaging.
    gemm_by_kernel = {}
    for e in raw:
        if not (e[3] and e[3].startswith(_GEMM_PREFIXES)):
            continue
        key = e[2].split('<')[0][:48]
        slot = gemm_by_kernel.setdefault(key, [0, 0.0])
        slot[0] += 1
        slot[1] += e[1]
    gemm_by_kernel = dict(sorted(((k, dict(n=v[0], ms=float(v[1])))
                                  for k, v in gemm_by_kernel.items()),
                                 key=lambda kv: -kv[1]['n'])[:12])

    # Busy time either side of each candidate phase split, so the
    # propagator/post-processing plots need no timeline replay.
    def split_busy(cut):
        if cut != cut:                                   # NaN
            return dict(before_ms=float('nan'), after_ms=float('nan'))
        before = sum(e[1] for e in raw if (e[0] - t0) / 1e6 < cut)
        return dict(before_ms=float(before), after_ms=float(busy_ms - before))

    # Exact per-phase split, when the caller supplies the compiler's own
    # instruction->phase map. Unlike the timestamp boundary this needs no
    # assumption about which op marks the handover, and it copes with ops such
    # as `dot_general` that legitimately occur in both phases.
    phases = {}
    if pmap:
        acc, unknown = {}, [0, 0.0]
        for s, dur, n, op in raw:
            ph = phase_of(op, pmap) if op else None
            if ph is None:
                unknown[0] += 1
                unknown[1] += dur
                continue
            slot = acc.setdefault(ph, {'n': 0, 'ms': 0.0, 'by_category': defaultdict(float)})
            slot['n'] += 1
            slot['ms'] += dur
            slot['by_category'][categorize(n)] += dur
        phases = {k: dict(n=v['n'], ms=float(v['ms']),
                          by_category={c: float(m) for c, m in v['by_category'].items()})
                  for k, v in acc.items()}
        phases['unassigned'] = dict(n=unknown[0], ms=float(unknown[1]), by_category={})

    return dict(
        phases=phases,
        n_kernels=len(raw),
        span_ms=float(span_ms),
        busy_ms=float(busy_ms),
        idle_ms=float(span_ms - busy_ms),
        n_gaps_gt1ms=int(len(big)),
        gaps_gt1ms_ms=float(big.sum()),
        median_gap_us=float(np.median(gaps) * 1000),
        gemm=agg(lambda e: e[3] is not None and e[3].startswith(_GEMM_PREFIXES)),
        diag=diag_stats,
        # where the propagator/post-processing split lands, for the phase plots
        first_gemm_ms=first_start(_GEMM_PREFIXES),
        first_post_ms=first_start(POST),
        split_at_gemm=split_busy(first_start(_GEMM_PREFIXES)),
        split_at_post=split_busy(first_start(POST)),
        by_op=by_op,
        by_kernel=by_kernel,
        gemm_by_kernel=gemm_by_kernel,
    )


# ---------------------------------------------------------------------------
# Independent cross-check via xprof
# ---------------------------------------------------------------------------

def kernel_totals_via_xprof(xplane_path):
    """Total GPU kernel time (ms) per kernel name, computed by xprof itself.

    `_stream_events` above reconstructs kernel time by walking the xplane by
    hand, which is only as good as its guesses about plane and line naming.
    xprof (already a dependency - it backs `uv run xprof --logdir ...`) computes
    the same aggregate with the code that powers its own UI, so disagreement
    between the two means our walk missed a stream.

    This CANNOT replace `phase_breakdown`: `KernelReport` carries `name`,
    `op_name`, `occurrences` and `total_duration_ns` but no timestamps, so there
    is no way to split the timeline at the post-processing boundary from it.
    It is a check on the totals, not a substitute for the phase split.

    Returns {kernel_name: ms}, or None if xprof produced nothing usable (it
    returns an empty result for CPU-backend traces, where deviceType is
    CPU_ONLY and there are no kernels at all).
    """
    try:
        from xprof.convert import raw_to_tool_data
    except ImportError:
        return None

    try:
        data, _ = raw_to_tool_data.xspace_to_tool_data([xplane_path], 'kernel_stats', {})
    except Exception as exc:  # xprof surfaces parse failures as arbitrary exceptions
        warnings.warn(f'xprof cross-check unavailable for {xplane_path}: {exc!r}',
                      RuntimeWarning, stacklevel=2)
        return None

    if isinstance(data, (str, bytes)):
        import json
        try:
            data = json.loads(data)
        except ValueError:
            return None

    # Tolerate both the {'data': [...]} table shape and the raw proto-JSON
    # {'reports': [...]}, and both snake_case and camelCase field spellings.
    rows = None
    if isinstance(data, dict):
        rows = data.get('data') or data.get('reports')
    if not rows:
        return None

    totals = defaultdict(float)
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get('name') or row.get('kernelName')
        ns = row.get('total_duration_ns', row.get('totalDurationNs'))
        if name is None or ns is None:
            continue
        totals[name] += float(ns) / 1e6
    return dict(totals) or None


def _cross_check(xplane_path, diagnostics, tolerance=0.05):
    """Warn if our hand-rolled stream walk disagrees with xprof's kernel totals."""
    xprof_totals = kernel_totals_via_xprof(xplane_path)
    if not xprof_totals:
        return None
    xprof_ms = sum(xprof_totals.values())
    ours_ms = diagnostics['propagator_ms'] + diagnostics['post_ms']
    if xprof_ms <= 0:
        return None
    rel = abs(ours_ms - xprof_ms) / xprof_ms
    if rel > tolerance:
        warnings.warn(
            f'Kernel-time mismatch in {xplane_path}: this module walked '
            f'{ours_ms:.1f} ms of stream events, xprof reports {xprof_ms:.1f} ms '
            f'({100 * rel:.0f}% apart). A stream line was probably missed.',
            RuntimeWarning, stacklevel=2)
    return xprof_ms


def build(trace_root, solver, dims, runs, strict=True, cross_check=True):
    """Average `phase_breakdown` over `runs`, for each dimension in `dims`."""
    result = {}
    for dim in dims:
        accum = defaultdict(list)
        diags = []
        missing = []
        for r in runs:
            trace_dir = f'{trace_root}/{solver}_d{dim}_run{r}'
            xplane_path = _find_xplane(trace_dir)
            if xplane_path is None:
                missing.append(r)
                continue
            cats, diag = phase_breakdown(xplane_path, strict=strict)
            if cross_check:
                _cross_check(xplane_path, diag)
            diags.append(diag)
            for k, v in cats.items():
                accum[k].append(v)

        if missing:
            # Previously silent, which made a partly-failed SLURM array look like
            # a complete result with slightly noisier means.
            print(f'd={dim}: WARNING no trace for run(s) {missing} '
                  f'- averaging over {len(diags)} of {len(runs)}')
        if not accum:
            print(f'd={dim}: no usable traces, skipped')
            continue

        result[dim] = {k: float(np.mean(v)) for k, v in accum.items()}
        tot = sum(result[dim].values())
        post = float(np.mean([d['post_ms'] for d in diags]))
        other = float(np.mean([d['other_frac'] for d in diags]))
        print(f'd={dim}: propagator total {tot:.1f} ms '
              f'(post-processing {post:.1f} ms, uncategorised {100 * other:.0f}%)')
        for k, v in sorted(result[dim].items(), key=lambda x: -x[1]):
            print(f'   {v:8.1f} ms  {100 * v / tot:5.1f}%  {k}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trace-root', default='out/gpu_h200_nocmdbuf/traces',
                        help='Directory of captured traces. Prefer a run without '
                             '+WHILE command buffers: with them the Tsit5 loop is one '
                             'CUDA graph and per-kernel attribution is lost.')
    parser.add_argument('--solver', default='cayley')
    parser.add_argument('--dims', type=int, nargs='+',
                        default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument('--runs', type=int, nargs='+', default=list(range(10)),
                        help='Run indices to average over (default: match submit.sh)')
    parser.add_argument('--output', default=None,
                        help='Default: <trace-root>/../propagator_breakdown.npy, so '
                             'the breakdown lands beside the traces it came from.')
    parser.add_argument('--no-strict', action='store_true',
                        help='Downgrade the uncategorised-kernel check to a warning')
    parser.add_argument('--no-cross-check', action='store_true',
                        help='Skip validating stream totals against xprof')
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(args.trace_root.rstrip('/')), 'propagator_breakdown.npy')

    result = build(args.trace_root, args.solver, args.dims, args.runs,
                   strict=not args.no_strict, cross_check=not args.no_cross_check)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.save(output, result, allow_pickle=True)
    print(f'\nsaved {output}')
