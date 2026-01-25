from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class Plan:
    plan_id: str
    L_bytes: int
    Theta_ms: float
    delta_L: int
    delta_Theta_ms: float


@dataclass
class RunResult:
    accepted: bool
    reject_reason: str  # "" if accepted
    n_real: int
    n_covers: int
    n_total: int
    Hmin: float
    L_base: int
    pad_bytes: int
    L_final: int
    theta_base_ms: float
    theta_final_ms: float
    drift_ms: float


def hmin_from_counts(counts: Dict[str, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    q2 = 0.0
    for c in counts.values():
        q = c / n
        q2 += q * q
    if q2 <= 0:
        return 0.0
    meff = 1.0 / q2
    return math.log(meff, 2)


def top_k_signatures(sig_list: List[str], k: int) -> List[str]:
    # stable top-k by frequency then lexicographic for determinism
    freq: Dict[str, int] = {}
    for s in sig_list:
        freq[s] = freq.get(s, 0) + 1
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [s for s, _ in items[: max(1, min(k, len(items)) )]]


def synthesize_ring(
    user_ids: np.ndarray,
    sigs: np.ndarray,
    rng: np.random.Generator,
    nmin: int,
    H_star: float,
    delta_n_max: int,
    k_support: int,
) -> Tuple[int, int, int, float, Dict[str, int]]:
    """Return (n_real, n_covers, n_total, Hmin, bucket_counts)."""

    if len(user_ids) < max(3, nmin):
        return 0, 0, 0, 0.0, {}

    # pick signer
    signer_idx = int(rng.integers(0, len(user_ids)))
    signer_id = user_ids[signer_idx]

    # target support S0 = top-k signatures
    S0 = top_k_signatures(list(sigs), k_support)
    S0_set = set(S0)

    # start counts with signer
    signer_sig = str(sigs[signer_idx])
    counts: Dict[str, int] = {}
    counts[signer_sig] = 1

    chosen = {signer_id}

    # desired counts for uniform q* over S0
    k = len(S0)
    base = nmin // k
    rem = nmin % k
    desired: Dict[str, int] = {s: base for s in S0}
    for s in S0[:rem]:
        desired[s] += 1

    # deficits include signer contribution
    deficits: Dict[str, int] = {}
    for s in S0:
        deficits[s] = max(0, desired[s] - counts.get(s, 0))

    # index pool per signature
    pools: Dict[str, List[int]] = {s: [] for s in S0}
    for i, (uid, sg) in enumerate(zip(user_ids, sigs)):
        if uid in chosen:
            continue
        sg = str(sg)
        if sg in pools:
            pools[sg].append(i)

    # fill ring to nmin
    n_needed = nmin - 1
    while n_needed > 0:
        # pick sig with highest deficit that has available pool
        candidates = [s for s in S0 if deficits.get(s, 0) > 0 and pools.get(s)]
        if candidates:
            # highest deficit first; break ties deterministically
            candidates.sort(key=lambda s: (-deficits[s], s))
            s_pick = candidates[0]
            i_pick = pools[s_pick].pop()
        else:
            # fallback: pick any remaining user
            remaining = [i for i, uid in enumerate(user_ids) if uid not in chosen]
            if not remaining:
                break
            i_pick = int(rng.choice(remaining))
            s_pick = str(sigs[i_pick])

        uid = user_ids[i_pick]
        if uid in chosen:
            continue
        chosen.add(uid)
        counts[s_pick] = counts.get(s_pick, 0) + 1
        if s_pick in deficits:
            deficits[s_pick] = max(0, deficits[s_pick] - 1)
        n_needed -= 1

    n_real = len(chosen)

    # PadShape: add cover keys to raise Hmin to H_star by equalizing counts over S0
    # Covers add to the currently smallest-count bucket in S0.
    n_covers = 0
    H = hmin_from_counts({s: counts.get(s, 0) for s in S0_set})

    while H < H_star and n_covers < delta_n_max:
        # choose bucket with smallest count (tie -> lexicographic)
        bucket = min(S0, key=lambda s: (counts.get(s, 0), s))
        counts[bucket] = counts.get(bucket, 0) + 1
        n_covers += 1
        H = hmin_from_counts({s: counts.get(s, 0) for s in S0_set})

    n_total = n_real + n_covers
    return n_real, n_covers, n_total, H, counts


def simulate_shape(plan: Plan, n_total: int, rng: np.random.Generator) -> Tuple[bool, str, int, int, float, float, float]:
    """Return (accepted, reason, L_base, pad_bytes, theta_base, theta_final, drift)."""

    # transcript model: fixed header + AOV proof + per-member component + small noise
    hdr = 256
    proof = 512
    per_member = 48
    noise = int(rng.integers(-16, 17))
    L_base = hdr + proof + per_member * n_total + noise

    # time model (ms): fixed overhead + per-member + AOV + small noise
    t_over = 2.0
    t_aov = 1.0
    t_per = 0.05
    base_noise = float(rng.uniform(0.0, 0.10))
    theta_base = t_over + t_aov + t_per * n_total + base_noise

    # cannot shrink length or time
    if L_base > plan.L_bytes + plan.delta_L:
        return False, "length", L_base, 0, theta_base, 0.0, 0.0
    if theta_base > plan.Theta_ms + plan.delta_Theta_ms:
        return False, "time", L_base, 0, theta_base, 0.0, 0.0

    pad_bytes = max(0, plan.L_bytes - L_base)
    L_final = L_base + pad_bytes

    dummy = max(0.0, plan.Theta_ms - theta_base)
    # residual jitter after dummy steps
    eta = float(rng.uniform(-plan.delta_Theta_ms / 2.0, plan.delta_Theta_ms / 2.0))
    theta_final = theta_base + dummy + eta

    drift = abs(theta_final - plan.Theta_ms)

    # enforce checks (what verifier does)
    if abs(L_final - plan.L_bytes) > plan.delta_L:
        return False, "length", L_base, pad_bytes, theta_base, theta_final, drift
    if abs(theta_final - plan.Theta_ms) > plan.delta_Theta_ms:
        return False, "time", L_base, pad_bytes, theta_base, theta_final, drift

    return True, "", L_base, pad_bytes, theta_base, theta_final, drift
