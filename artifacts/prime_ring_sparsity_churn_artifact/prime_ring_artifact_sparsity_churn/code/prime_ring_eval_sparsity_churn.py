#!/usr/bin/env python3
"""PRIME-Ring evaluation: ring synthesis under sparsity and churn.

This script runs a lightweight version of the PRIME-Ring synthesis pipeline
(Profile/Target/Synthesize/PadShape/Certify) on the generated E-health and
OSN datasets, then produces reproducible summary tables.

Outputs
  out_dir/
    - results_raw.csv           per-run records
    - results_agg.csv           aggregated by (domain,tau,H_star,d0,churn)
    - table_sparsity_churn_summary.tex
    - table_sparsity_churn_hardcase.tex

Run
  python prime_ring_eval_sparsity_churn.py --out_dir /mnt/data/prime_ring_sparsity_churn --seed 0

Inputs (already present in this workspace)
  E-health:
    - /mnt/data/prime_ring_ehealth_access_log.csv
    - /mnt/data/prime_ring_ehealth_staff_graph_edges.csv
  OSN:
    - /mnt/data/prime_ring_osn_twitter_users.csv
    - /mnt/data/prime_ring_osn_twitter_edges.csv
    - /mnt/data/prime_ring_osn_twitter_actions.csv
    - /mnt/data/prime_ring_osn_facebook_users.csv
    - /mnt/data/prime_ring_osn_facebook_edges.csv
    - /mnt/data/prime_ring_osn_facebook_actions.csv

Notes
  - d0 is implemented as an exclusion of (d0-1)-hop neighborhoods (fast).
  - churn randomly drops candidates at sampling time with probability churn.
  - cover keys are synthetic decoys labelled by buckets; they fill deficits and
    are used to equalize exposure up to a budget.
"""

from __future__ import annotations

import argparse
import os
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

# Default data directory (overridden by CLI)
DATA_DIR = './data'

import pandas as pd


# -----------------------------
# Metrics
# -----------------------------

def h_min_from_counts(counts: Dict[str, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return float("nan")
    c = 0.0
    for v in counts.values():
        q = v / n
        c += q * q
    if c <= 0:
        return float("nan")
    m_eff = 1.0 / c
    return float(math.log(m_eff, 2))


def tau_hat_from_ring(
    members: List[Tuple[str, str, int]],
    bucket_counts: Dict[str, int],
    d0: int,
    gamma: float = 0.0,
) -> float:
    """Certified posterior bound (product prior proxy).

    members: list of (member_type, bucket, is_signer)
      - member_type: 'real' or 'cover'
      - bucket: attribute bucket label
      - is_signer: 1 if signer else 0

    For speed and determinism, we use a distance proxy:
      dist(signer)=0; dist(decoy)=d0 (since decoys are filtered to be >= d0).
    If gamma=0, distance term disappears (certificate driven by exposure only).
    """
    n = len(members)
    if n == 0:
        return float("nan")

    def pi(dist: int) -> float:
        # bounded prior, normalized not required for ratio
        if gamma <= 0:
            return 1.0
        return float(max(1e-6, math.exp(-gamma * dist)))

    weights = []
    for mtype, bucket, is_signer in members:
        q = bucket_counts.get(bucket, 0) / n
        if q <= 0:
            return float("nan")
        dist = 0 if is_signer else max(1, d0)
        w = (1.0 / q) * (1.0 / pi(dist))
        weights.append(w)

    s = sum(weights)
    if s <= 0:
        return float("nan")
    posteriors = [w / s for w in weights]
    return float(max(posteriors))


# -----------------------------
# Graph utilities (fast neighborhood exclusion)
# -----------------------------


def build_adj_from_edges(df: pd.DataFrame, src: str, dst: str, directed: bool = False) -> Dict[Any, List[Any]]:
    adj: Dict[Any, List[Any]] = defaultdict(list)
    s = df[src].to_numpy()
    t = df[dst].to_numpy()
    for a, b in zip(s, t):
        a = a.item() if hasattr(a, 'item') else a
        b = b.item() if hasattr(b, 'item') else b
        adj[a].append(b)
        if not directed:
            adj[b].append(a)
    return adj


def neighborhood_within_k(adj: Dict[Any, List[Any]], start: Any, k: int) -> Set[Any]:
    """Return nodes within <= k hops (including start). k is small (0..2)."""
    if k <= 0:
        return {start}
    seen = {start}
    frontier = {start}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return seen


# -----------------------------
# Domain configs
# -----------------------------


@dataclass
class DomainData:
    name: str
    events: pd.DataFrame
    id_col: str
    bucket_col: str
    epoch_col: str
    adj: Dict[Any, List[Any]]
    bucket_nodes: Dict[str, np.ndarray]
    bucket_order: List[str]


def make_bucket_nodes(ids: Iterable[Any], buckets: Iterable[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    tmp: Dict[str, List[Any]] = defaultdict(list)
    for u, b in zip(ids, buckets):
        u = u.item() if hasattr(u, 'item') else u
        tmp[str(b)].append(u)
    bucket_nodes: Dict[str, np.ndarray] = {}
    for b, v in tmp.items():
        try:
            bucket_nodes[b] = np.asarray(v, dtype=np.int64)
        except Exception:
            bucket_nodes[b] = np.asarray(v, dtype=object)
    bucket_order = sorted(bucket_nodes.keys(), key=lambda x: -len(bucket_nodes[x]))
    return bucket_nodes, bucket_order


def load_ehealth() -> DomainData:
    events = pd.read_csv(os.path.join(DATA_DIR, "prime_ring_ehealth_access_log.csv"))
    edges = pd.read_csv(os.path.join(DATA_DIR, "prime_ring_ehealth_staff_graph_edges.csv"))

    # Build per-staff bucket (mode)
    staff_bucket = events.groupby("staff_id")["attr_bucket"].agg(lambda s: s.value_counts().idxmax())
    ids = staff_bucket.index.astype(str).to_numpy()
    buckets = staff_bucket.astype(str).to_numpy()
    bucket_nodes, bucket_order = make_bucket_nodes(ids, buckets)

    adj = build_adj_from_edges(edges, src="src_staff_id", dst="dst_staff_id", directed=False)

    # Events: keep what we need
    ev = events[["staff_id", "attr_bucket", "epoch_t"]].copy()
    ev = ev.rename(columns={"attr_bucket": "bucket"})

    return DomainData(
        name="E-health",
        events=ev,
        id_col="staff_id",
        bucket_col="bucket",
        epoch_col="epoch_t",
        adj=adj,
        bucket_nodes=bucket_nodes,
        bucket_order=bucket_order,
    )


def load_osn(name: str) -> DomainData:
    base = "twitter" if "Twitter" in name else "facebook"
    users = pd.read_csv(os.path.join(DATA_DIR, f"prime_ring_osn_{base}_users.csv"))
    edges = pd.read_csv(os.path.join(DATA_DIR, f"prime_ring_osn_{base}_edges.csv"))
    actions = pd.read_csv(os.path.join(DATA_DIR, f"prime_ring_osn_{base}_actions.csv"))

    # bucket = attr_sig
    ids = users["user_id"].astype(int).to_numpy()
    buckets = users["attr_sig"].astype(str).to_numpy()
    bucket_nodes, bucket_order = make_bucket_nodes(ids, buckets)

    adj = build_adj_from_edges(edges, src="src_id", dst="dst_id", directed=False)

    ev = actions[["moderator_id", "epoch"]].copy()
    # attach bucket by join
    ev = ev.merge(users[["user_id", "attr_sig"]], left_on="moderator_id", right_on="user_id", how="left")
    ev = ev.rename(columns={"attr_sig": "bucket"})

    return DomainData(
        name=name,
        events=ev[["moderator_id", "bucket", "epoch"]].dropna(),
        id_col="moderator_id",
        bucket_col="bucket",
        epoch_col="epoch",
        adj=adj,
        bucket_nodes=bucket_nodes,
        bucket_order=bucket_order,
    )


# -----------------------------
# Synthesis + pad
# -----------------------------


def choose_support(bucket_order: List[str], signer_bucket: str, K: int) -> List[str]:
    if K <= 0:
        return []
    supp = bucket_order[:K]
    if signer_bucket in supp:
        return supp
    # replace last with signer bucket
    if len(supp) < K:
        supp = supp + [signer_bucket]
    else:
        supp = supp[:-1] + [signer_bucket]
    return supp


def desired_counts_uniform(n: int, support: List[str]) -> Dict[str, int]:
    K = len(support)
    if K == 0:
        return {}
    base = n // K
    rem = n - base * K
    counts = {b: base for b in support}
    # deterministic remainder allocation by support order
    for i in range(rem):
        counts[support[i % K]] += 1
    return counts


def sample_one(
    rng: np.random.Generator,
    arr: np.ndarray,
    forbidden: Set[Any],
    churn: float,
    max_tries: int = 50,
) -> int | None:
    if len(arr) == 0:
        return None
    for _ in range(max_tries):
        u = arr[rng.integers(0, len(arr))]
        u = u.item() if hasattr(u, "item") else u
        if u in forbidden:
            continue
        if churn > 0 and rng.random() < churn:
            continue
        return u
    return None


def synthesize_ring_one_run(
    dd: DomainData,
    signer_id: int,
    signer_bucket: str,
    tau: float,
    H_star: int,
    d0: int,
    churn: float,
    rng: np.random.Generator,
    delta_n_max: int = 200,
) -> Dict[str, float | int]:
    """One synthesis run, returns metrics and failure flags."""

    # Targets
    n = int(math.ceil(1.0 / tau))
    support_size = len(dd.bucket_order)
    K_need = int(math.ceil(2 ** H_star))
    K = min(support_size, K_need)

    infeasible_support = int(K < K_need)

    support = choose_support(dd.bucket_order, signer_bucket, K)
    if signer_bucket not in support:
        # should not happen, but guard
        support = support + [signer_bucket]

    desired = desired_counts_uniform(n, support)

    # Distance guard via neighborhood exclusion
    near = neighborhood_within_k(dd.adj, signer_id, max(0, d0 - 1))
    forbidden: Set[Any] = set(near)
    forbidden.add(signer_id)

    # Members: list of (type, bucket, is_signer)
    members: List[Tuple[str, str, int]] = [("real", str(signer_bucket), 1)]
    chosen_real: Set[Any] = {signer_id}

    empty_pool_hit = 0

    # Sample real decoys bucket-wise
    for b in support:
        need = desired.get(b, 0) - (1 if b == signer_bucket else 0)
        if need <= 0:
            continue
        arr = dd.bucket_nodes.get(b, np.asarray([], dtype=object))
        got = 0
        for _ in range(need):
            u = sample_one(rng, arr, forbidden | chosen_real, churn=churn)
            if u is None:
                empty_pool_hit = 1
                break
            chosen_real.add(u)
            members.append(("real", b, 0))
            got += 1
        # if we broke early, remaining deficit becomes covers

    # Fill to size n with cover keys
    cover_added = 0
    while len(members) < n and cover_added < delta_n_max:
        # choose a bucket that is currently under target
        counts_now = defaultdict(int)
        for _, bb, _ in members:
            counts_now[bb] += 1
        # compute deficit wrt desired
        best_b = None
        best_def = -10**9
        for b in support:
            d = desired.get(b, 0) - counts_now.get(b, 0)
            if d > best_def:
                best_def = d
                best_b = b
        if best_b is None:
            best_b = support[0]
        members.append(("cover", best_b, 0))
        cover_added += 1

    # Exposure counts
    bucket_counts = defaultdict(int)
    for _, bb, _ in members:
        bucket_counts[bb] += 1

    # PadShape: add cover keys to equalize, attempt to reach H*
    # H_min is bounded by log2 K; if K is too small, infeasible_support already flags.
    added_for_entropy = 0
    H_min = h_min_from_counts(bucket_counts)
    if not infeasible_support:
        # Greedy equalization
        while H_min < H_star and (cover_added + added_for_entropy) < delta_n_max:
            # add to smallest count bucket
            b_min = min(support, key=lambda b: bucket_counts.get(b, 0))
            members.append(("cover", b_min, 0))
            bucket_counts[b_min] += 1
            added_for_entropy += 1
            H_min = h_min_from_counts(bucket_counts)

    # Certificate (tau_hat)
    tau_hat = tau_hat_from_ring(members, bucket_counts, d0=d0, gamma=0.0)

    success = int((H_min >= H_star) and (tau_hat <= tau))

    # Refusal: either infeasible support or failed targets after padding
    refusal = int((not success) and (infeasible_support or (H_min < H_star) or (tau_hat > tau)))

    cover_total = sum(1 for t, _, _ in members if t == "cover")
    cover_frac = cover_total / len(members) if members else float("nan")

    return {
        "H_min": float(H_min),
        "tau_hat": float(tau_hat),
        "S_size": int(len(members)),
        "success": success,
        "refusal": refusal,
        "infeasible_support": infeasible_support,
        "empty_pool_hit": empty_pool_hit,
        "cover_frac": float(cover_frac),
    }


# -----------------------------
# Tables
# -----------------------------


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = df.columns.tolist()
    # first columns left, rest right
    left_n = 5 if len(cols) >= 5 else max(1, len(cols) - 3)
    colspec = "l" * left_n + "r" * (len(cols) - left_n)

    def fmt(x, col):
        if pd.isna(x):
            return "nan"
        if col in {"SR", "Refusal", "EmptyPool", "CoverFrac"}:
            return f"{float(x):.3f}"
        if col in {"H_min", "tau_hat", "S'"}:
            return f"{float(x):.3f}" if col != "S'" else f"{float(x):.1f}"
        return str(x)

    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt(r[c], c) for c in cols) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/mnt/data/prime_ring_sparsity_churn")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs_per_setting", type=int, default=60)
    args = ap.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir


    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    domains = [load_ehealth(), load_osn("OSN-Twitter"), load_osn("OSN-Facebook")]

    # Sweep grid (kept small enough to run quickly but covers the intended axes)
    taus = [0.05, 0.10, 0.20]
    H_stars = [3, 4, 5]
    d0s = [1, 2, 3]
    churns = [0.0, 0.2, 0.4]

    raw_rows = []

    for dd in domains:
        ev = dd.events.dropna(subset=[dd.id_col, dd.bucket_col]).copy()
        # Keep ID type (int for OSN, str for E-health).
        try:
            if pd.api.types.is_numeric_dtype(ev[dd.id_col]):
                ev[dd.id_col] = ev[dd.id_col].astype(int)
            else:
                ev[dd.id_col] = ev[dd.id_col].astype(str)
        except Exception:
            ev[dd.id_col] = ev[dd.id_col].astype(str)
        ev[dd.bucket_col] = ev[dd.bucket_col].astype(str)

        # Sample pool of events for reproducibility and speed
        # We sample with replacement per setting to keep per-setting distribution stable.
        ev_idx = np.arange(len(ev))

        for tau in taus:
            for Hs in H_stars:
                for d0 in d0s:
                    for churn in churns:
                        for r_i in range(args.runs_per_setting):
                            row = ev.iloc[int(rng.choice(ev_idx))]
                            signer_id = row[dd.id_col]
                            signer_bucket = str(row[dd.bucket_col])

                            m = synthesize_ring_one_run(
                                dd=dd,
                                signer_id=signer_id,
                                signer_bucket=signer_bucket,
                                tau=tau,
                                H_star=Hs,
                                d0=d0,
                                churn=churn,
                                rng=rng,
                            )

                            raw_rows.append(
                                {
                                    "Domain": dd.name,
                                    "tau": tau,
                                    "H_star": Hs,
                                    "d0": d0,
                                    "churn": churn,
                                    **m,
                                }
                            )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "results_raw.csv", index=False)

    agg = (
        raw.groupby(["Domain", "tau", "H_star", "d0", "churn"], as_index=False)
        .agg(
            H_min=("H_min", "mean"),
            tau_hat=("tau_hat", "mean"),
            SR=("success", "mean"),
            **{"S'": ("S_size", "mean")},
            Refusal=("refusal", "mean"),
            EmptyPool=("empty_pool_hit", "mean"),
            CoverFrac=("cover_frac", "mean"),
        )
        .sort_values(["Domain", "churn", "tau", "H_star", "d0"])
    )
    agg.to_csv(out_dir / "results_agg.csv", index=False)

    # Table 1: per-domain summary aggregated over the sweep for each churn
    summary = (
        agg.groupby(["Domain", "churn"], as_index=False)
        .agg(
            H_min=("H_min", "mean"),
            tau_hat=("tau_hat", "mean"),
            SR=("SR", "mean"),
            **{"S'": ("S'", "mean")},
            Refusal=("Refusal", "mean"),
            EmptyPool=("EmptyPool", "mean"),
            CoverFrac=("CoverFrac", "mean"),
        )
        .sort_values(["Domain", "churn"])
    )
    t1 = latex_table(
        summary,
        caption=(
            "Ring synthesis under sparsity and churn. Values are averages over a sweep of (tau, H^\\star, d_0) "
            "and repeated runs per setting. SR is the fraction of runs meeting (H_{min}\\ge H^\\star) and (\\hat{\\tau}\\le tau)."
        ),
        label="tab:prime_sparsity_churn_summary",
    )
    (out_dir / "table_sparsity_churn_summary.tex").write_text(t1)

    # Table 2: hard case (tight tau, high H*, larger d0) across churn
    hard = agg[(agg["tau"] == 0.05) & (agg["H_star"] == 5) & (agg["d0"] == 3)].copy()
    hard = hard[["Domain", "churn", "H_min", "tau_hat", "SR", "S'", "Refusal", "EmptyPool", "CoverFrac"]].sort_values(
        ["Domain", "churn"]
    )
    t2 = latex_table(
        hard,
        caption=(
            "Hard-case ring synthesis (tau=0.05, H^\\star=5, d_0=3) across churn. "
            "EmptyPool is the fraction of runs where some target bucket had no distance-safe candidates after churn; CoverFrac is the mean cover-key share."
        ),
        label="tab:prime_sparsity_churn_hardcase",
    )
    (out_dir / "table_sparsity_churn_hardcase.tex").write_text(t2)

    print("Wrote outputs to", out_dir)


if __name__ == "__main__":
    main()
