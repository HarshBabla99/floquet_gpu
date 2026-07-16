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


def summarize_trace(trace_dir, top_k=15):
    """Aggregate per-HLO-op device time from a captured trace.

    Events tagged with an 'hlo_op' stat are compiled-XLA execution on a device
    (this holds regardless of backend/plane naming); everything else is host-side
    (Python dispatch, tracing, etc.) and is left to the interactive trace viewer,
    since the host trace is a nested call stack and can't be summed flatly.

    Returns a dict with:
      device_op_ns:    {op_name: total_ns} over the whole trace
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
            for event in line.events:
                stats = dict(event.stats)
                op_name = stats.get('hlo_op')
                if op_name is None:
                    continue
                device_op_ns[op_name] = device_op_ns.get(op_name, 0.0) + event.duration_ns

    device_total_ns = sum(device_op_ns.values())
    top_ops = sorted(device_op_ns.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    return dict(
        device_op_ns=device_op_ns,
        device_total_ns=device_total_ns,
        top_ops=top_ops,
        xplane_path=xplane_path,
    )
