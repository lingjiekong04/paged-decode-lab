"""Host-only measurement policy; not kernel tuning or backend selection."""
import math

UNROLLS=(1,8,32,128)
CONVERGENCE_LIMIT=.05


def interval_per_operator(elapsed_ms,replays,unroll):
    if not math.isfinite(elapsed_ms) or elapsed_ms<=0:
        raise ValueError('positive finite elapsed interval required')
    if type(replays) is not int or replays<1 or type(unroll) is not int or unroll<1:
        raise ValueError('positive integer replay/unroll counts required')
    return elapsed_ms*1000/(replays*unroll)


def convergence_delta(t32,t128):
    if min(t32,t128)<=0 or not all(map(math.isfinite,(t32,t128))):
        raise ValueError('positive finite timings required')
    return abs(t32-t128)/t128
