import glob
import os

import jax


def _find_xplane(trace_dir):
    """Locate the .xplane.pb file jax.profiler.trace() writes under trace_dir."""
    matches = glob.glob(os.path.join(trace_dir, 'plugins', 'profile', '*', '*.xplane.pb'))
    if not matches:
        return None
    # trace() writes exactly one xspace file per call; if more exist, take the newest.
    return max(matches, key=os.path.getmtime)


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


def summarize_trace(trace_dir, top_k=15):
    """Aggregate per-HLO-op device time from a captured trace.

    Events tagged with an 'hlo_op' stat are compiled-XLA execution on a device
    (this holds regardless of backend/plane naming); everything else is host-side
    (Python dispatch, tracing, etc.) and is left to the interactive trace viewer.
    Within a plane/line, hlo_op events can nest (see `_self_times_ns`), so
    per-op time is the *self* time, not raw event duration - otherwise e.g. a
    `while` loop's time would be counted once for the loop and again for every
    op inside its body.

    Returns a dict with:
      device_op_ns:    {op_name: total_self_ns} over the whole trace
      device_total_ns: sum of device_op_ns.values()
      top_ops:         top_k (op_name, ns) pairs, sorted descending
      xplane_path:     path to the underlying .xplane.pb, or None if not found
    """
    xplane_path = _find_xplane(trace_dir)
    if xplane_path is None:
        return dict(device_op_ns={}, device_total_ns=0, top_ops=[], xplane_path=None)

    profile = jax.profiler.ProfileData.from_file(xplane_path)

    device_op_ns = {}
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
                device_op_ns[op_name] = device_op_ns.get(op_name, 0.0) + self_ns

    device_total_ns = sum(device_op_ns.values())
    top_ops = sorted(device_op_ns.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    return dict(
        device_op_ns=device_op_ns,
        device_total_ns=device_total_ns,
        top_ops=top_ops,
        xplane_path=xplane_path,
    )
