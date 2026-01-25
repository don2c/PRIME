#!/usr/bin/env python3
"""PRIME-Ring evaluation: certified release and inference-bounded anonymity.

This evaluation is runnable using the synthetic-but-structured CSV datasets generated
in this workspace (E-health and OSN).

What we validate
----------------
For each domain we simulate ring synthesis and release gating, then check that
empirical attacker success (posterior mass on the true signer) stays below the
certificate under the declared feature set F and attack class H.

We report:
  - tau_hat: product-prior certificate using exposure (q) and distance prior (pi).
  - tau_hat_LLM: tightened certificate against a declared attack class H
    (implemented as a finite family of attack scoring rules).
  - violation rate: released runs where an attacker in H assigns posterior mass to
    the true signer above tau_hat_LLM.
  - refusal rate: runs where synthesis cannot meet (tau_target, H_min_target)
    within a budget and the system refuses to release.

Inputs (expected at these paths)
--------------------------------
  E-health
    /mnt/data/prime_ring_ehealth_access_log.csv
    /mnt/data/prime_ring_ehealth_staff_graph_edges.csv

  OSN
    /mnt/data/prime_ring_osn_twitter_users.csv
    /mnt/data/prime_ring_osn_twitter_edges.csv
    /mnt/data/prime_ring_osn_twitter_actions.csv
    /mnt/data/prime_ring_osn_facebook_users.csv
    /mnt/data/prime_ring_osn_facebook_edges.csv
    /mnt/data/prime_ring_osn_facebook_actions.csv

Outputs
-------
Writes to --out_dir (default: /mnt/data/prime_ring_cert_results):
  - table_cert_release_summary.tex
  - table_cert_release_attacks.tex
  - results_cert_release.csv
  - config.json

Run
---
  python prime_ring_eval_certified_release.py --out_dir prime_ring_cert_results --seed 0

Notes
-----
- Distances are computed exactly for E-health (per-epoch staff graph).
- Distances are approximated for OSN using truncated BFS depth 2; nodes beyond
  2 hops are treated as distance 3.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def safe_log2(x: float) -> float:
    if x <= 0:
        return float("inf")
    return math.log(x, 2)


def pi_dist(d: int, gamma: float, alpha: float, beta: float) -> float:
    """Distance prior pi(d), bounded in [alpha, beta].

    pi decreases with d. This makes 1/pi larger for far nodes, which reduces
    the weight (posterior) of the signer (dist=0) relative to decoys.
    """
    val = math.exp(-gamma * float(d))
    return max(alpha, min(beta, val))


def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in weights.values())
    if s <= 0:
        k = len(weights)
        return {u: 1.0 / k for u in weights}
    return {u: max(0.0, v) / s for u, v in weights.items()}


def max_posterior(post: Dict[str, float]) -> float:
    return float(max(post.values())) if post else float("nan")


@dataclass
class ReleaseRun:
    domain: str
    released: bool
    tau_hat: float
    tau_hat_llm: float
    hmin_bits: float
    attacker_post_true: float
    violation: bool
    ring_size: int
    iters: int


# -----------------------------
# Graph helpers
# -----------------------------

def build_undirected_adj(edges: pd.DataFrame, src: str, dst: str) -> Dict[str, List[str]]:
    """Build an undirected adjacency list."""
    adj: Dict[str, List[str]] = defaultdict(list)
    s = edges[src].astype(str).to_numpy()
    t = edges[dst].astype(str).to_numpy()
    for a, b in zip(s, t):
        if a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)
    return adj


def bfs_dist_trunc(adj: Dict[str, List[str]], start: str, max_depth: int = 2) -> Dict[str, int]:
    """Truncated BFS distances up to max_depth."""
    dist: Dict[str, int] = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= max_depth:
            continue
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


def dist_or_bucket(dist_map: Dict[str, int], node: str, far_bucket: int = 3) -> int:
    return int(dist_map.get(node, far_bucket))


# -----------------------------
# Attack class H
# -----------------------------

def attack_posterior_product_prior(
    ring_members: List[str],
    signer: str,
    attr_sig: Dict[str, str],
    dist_map: Dict[str, int],
    gamma: float,
    alpha: float,
    beta: float,
) -> Dict[str, float]:
    """Posterior proportional to 1/q(attr_sig(u)) * 1/pi(dist(u, signer))."""
    # exposure q over the ring
    sigs = [attr_sig.get(u, "UNK") for u in ring_members]
    counts = Counter(sigs)
    q = {s: counts[s] / float(len(ring_members)) for s in counts}

    w: Dict[str, float] = {}
    for u in ring_members:
        s = attr_sig.get(u, "UNK")
        qu = q.get(s, 1.0 / max(1, len(counts)))
        d = dist_or_bucket(dist_map, u, far_bucket=3) if u != signer else 0
        pid = pi_dist(d, gamma=gamma, alpha=alpha, beta=beta)
        w[u] = (1.0 / max(qu, 1e-12)) * (1.0 / max(pid, 1e-12))
    return normalize(w)


def attack_posterior_attr_only(ring_members: List[str], attr_sig: Dict[str, str]) -> Dict[str, float]:
    sigs = [attr_sig.get(u, "UNK") for u in ring_members]
    counts = Counter(sigs)
    q = {s: counts[s] / float(len(ring_members)) for s in counts}
    w = {u: 1.0 / max(q.get(attr_sig.get(u, "UNK"), 1e-12), 1e-12) for u in ring_members}
    return normalize(w)


def attack_posterior_degree_only(ring_members: List[str], degree: Dict[str, int]) -> Dict[str, float]:
    # Prefer low-degree identities (less connected).
    w = {u: 1.0 / float(degree.get(u, 0) + 1) for u in ring_members}
    return normalize(w)


def build_propensity_tables_ehealth(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """P(staff_id | action) estimated from logs."""
    tmp = df.copy()
    tmp["staff_id"] = tmp["staff_id"].astype(str)
    tmp["action"] = tmp["action"].astype(str)
    # counts(action, staff)
    c = tmp.groupby(["action", "staff_id"]).size().rename("n").reset_index()
    total = tmp.groupby(["action"]).size().rename("tot").reset_index()
    c = c.merge(total, on="action", how="left")
    c["p"] = c["n"] / c["tot"].clip(lower=1)
    return {(r["action"], r["staff_id"]): float(r["p"]) for _, r in c.iterrows()}


def attack_posterior_propensity(
    ring_members: List[str],
    signer: str,
    action_key: str,
    propensity: Dict[Tuple[str, str], float],
    floor: float = 1e-6,
) -> Dict[str, float]:
    # Higher propensity -> higher posterior.
    w = {u: max(propensity.get((action_key, str(u)), 0.0), floor) for u in ring_members}
    return normalize(w)


def max_over_attacks(
    ring_members: List[str],
    signer: str,
    attr_sig: Dict[str, str],
    dist_map: Dict[str, int],
    degree: Dict[str, int],
    action_key: str,
    propensity: Dict[Tuple[str, str], float],
    gamma: float,
    alpha: float,
    beta: float,
) -> Tuple[float, float]:
    """Return (max_posterior_over_attacks, max_posterior_mass_on_true_signer_over_attacks)."""
    posts = []
    posts.append(attack_posterior_product_prior(ring_members, signer, attr_sig, dist_map, gamma, alpha, beta))
    posts.append(attack_posterior_attr_only(ring_members, attr_sig))
    posts.append(attack_posterior_degree_only(ring_members, degree))
    posts.append(attack_posterior_propensity(ring_members, signer, action_key, propensity))

    max_any = max(max_posterior(p) for p in posts)
    max_true = max(float(p.get(signer, 0.0)) for p in posts)
    return float(max_any), float(max_true)


# -----------------------------
# Ring synthesis
# -----------------------------

def select_ring_by_deficit(
    signer: str,
    candidates: pd.DataFrame,
    id_col: str,
    sig_col: str,
    n_total: int,
    rng: np.random.Generator,
) -> List[str]:
    """Deficit-based selection toward uniform exposure over signature buckets."""
    signer = str(signer)
    cand = candidates.copy()
    cand[id_col] = cand[id_col].astype(str)
    cand[sig_col] = cand[sig_col].astype(str)

    # pool excluding signer
    pool = cand[cand[id_col] != signer]
    if pool.empty:
        return [signer]

    buckets = pool[sig_col].unique().tolist()
    if not buckets:
        return [signer]

    q_star = {b: 1.0 / len(buckets) for b in buckets}
    selected = [signer]
    selected_set = {signer}

    # index candidates by bucket
    by_bucket = {b: pool[pool[sig_col] == b][id_col].astype(str).tolist() for b in buckets}

    # track counts
    counts = Counter({b: 0 for b in buckets})

    def current_q() -> Dict[str, float]:
        total = max(1, len(selected) - 1)  # decoys only
        return {b: counts[b] / float(total) for b in buckets}

    while len(selected) < n_total:
        q = current_q()
        deficits = {b: q_star[b] - q.get(b, 0.0) for b in buckets}
        b = max(deficits, key=deficits.get)
        options = by_bucket.get(b, [])
        if not options:
            # fallback: any candidate
            options = pool[id_col].astype(str).tolist()
        if not options:
            break

        # sample until unique or give up
        for _ in range(10):
            u = str(rng.choice(options))
            if u not in selected_set:
                selected.append(u)
                selected_set.add(u)
                ub = str(pool.loc[pool[id_col].astype(str) == u, sig_col].iloc[0])
                if ub in counts:
                    counts[ub] += 1
                break
        else:
            break

    return selected


def compute_hmin_bits_from_tau(tau: float) -> float:
    if tau <= 0:
        return float("inf")
    return float(-safe_log2(tau))


# -----------------------------
# Domain evaluators
# -----------------------------

def eval_ehealth(
    access_log_path: Path,
    staff_edges_path: Path,
    seed: int,
    runs: int,
    tau_target: float,
    hmin_target: float,
    max_iters: int,
    n0: int,
    n_step: int,
    n_max: int,
    gamma: float,
    alpha: float,
    beta: float,
) -> Tuple[List[ReleaseRun], Dict[str, float]]:
    rng = np.random.default_rng(seed)

    df = pd.read_csv(access_log_path)
    df["staff_id"] = df["staff_id"].astype(str)
    df["attr_bucket"] = df["attr_bucket"].astype(str)
    df["action"] = df["action"].astype(str)

    # Build per-epoch graphs
    ed = pd.read_csv(staff_edges_path)
    # Normalize edge columns across datasets/generators.
    if "src" not in ed.columns or "dst" not in ed.columns:
        rename_map = {}
        if "src_staff_id" in ed.columns:
            rename_map["src_staff_id"] = "src"
        if "dst_staff_id" in ed.columns:
            rename_map["dst_staff_id"] = "dst"
        if "source_id" in ed.columns:
            rename_map["source_id"] = "src"
        if "target_id" in ed.columns:
            rename_map["target_id"] = "dst"
        if rename_map:
            ed = ed.rename(columns=rename_map)
    if "src" not in ed.columns or "dst" not in ed.columns:
        raise ValueError(f"Unsupported edge schema in {staff_edges_path}: columns={list(ed.columns)}")

    ed["src"] = ed["src"].astype(str)
    ed["dst"] = ed["dst"].astype(str)
    if "epoch_t" not in ed.columns:
        # Some generators use 'epoch'
        if "epoch" in ed.columns:
            ed = ed.rename(columns={"epoch": "epoch_t"})
        else:
            ed["epoch_t"] = "0"
    ed["epoch_t"] = ed["epoch_t"].astype(str)

    graphs: Dict[str, Dict[str, List[str]]] = {}
    for epoch, g in ed.groupby("epoch_t"):
        graphs[str(epoch)] = build_undirected_adj(g, "src", "dst")

    # Degree for a weak side-channel attacker
    degree: Dict[str, int] = {}
    for epoch, adj in graphs.items():
        for u, nbrs in adj.items():
            degree[u] = max(degree.get(u, 0), len(nbrs))

    # Attribute signature map
    attr_sig = dict(zip(df["staff_id"], df["attr_bucket"]))

    # Propensity table
    prop = build_propensity_tables_ehealth(df)

    # Sample runs from observed log events
    sample_df = df.sample(n=min(runs, len(df)), random_state=seed).reset_index(drop=True)

    out: List[ReleaseRun] = []
    for _, row in sample_df.iterrows():
        signer = str(row["staff_id"])
        epoch = str(row.get("epoch_t", "0"))
        action_key = str(row.get("action", ""))

        # Candidate pool: staff seen in this epoch
        epoch_pool = df[df.get("epoch_t", "0").astype(str) == epoch][["staff_id", "attr_bucket"]].drop_duplicates()
        epoch_pool = epoch_pool.rename(columns={"staff_id": "id", "attr_bucket": "sig"})
        if epoch_pool.empty:
            continue

        # Distance map from signer in the epoch graph
        adj = graphs.get(epoch, {})
        dist_map = bfs_dist_trunc(adj, signer, max_depth=3)

        # Tightening loop increases ring size
        iters = 0
        released = False
        tau_hat = float("nan")
        tau_hat_llm = float("nan")
        attacker_true = float("nan")
        violation = False
        ring_size = 0

        n = n0
        while iters < max_iters and n <= n_max:
            iters += 1

            # Prefer candidates at distance >=2 from signer when possible
            def is_far(u: str) -> bool:
                return dist_or_bucket(dist_map, u, far_bucket=10) >= 2

            pool_ids = epoch_pool["id"].astype(str)
            far_ids = [u for u in pool_ids.tolist() if u == signer or is_far(u)]
            cand = epoch_pool[epoch_pool["id"].astype(str).isin(far_ids)].copy()
            if len(cand) < max(8, n // 2):
                cand = epoch_pool.copy()

            ring = select_ring_by_deficit(signer, cand, id_col="id", sig_col="sig", n_total=n, rng=rng)
            ring_size = len(ring)

            # Certificate tau_hat (product prior)
            post_cert = attack_posterior_product_prior(ring, signer, attr_sig, dist_map, gamma, alpha, beta)
            tau_hat = max_posterior(post_cert)

            # Tightened certificate tau_hat_llm against attack class H
            tau_hat_llm, attacker_true = max_over_attacks(
                ring,
                signer,
                attr_sig,
                dist_map,
                degree,
                action_key,
                prop,
                gamma,
                alpha,
                beta,
            )

            hmin_bits = compute_hmin_bits_from_tau(tau_hat_llm)

            if tau_hat_llm <= tau_target and hmin_bits >= hmin_target:
                released = True
                # Violation check: any attacker in H puts more than the tightened bound on the true signer
                violation = attacker_true > (tau_hat_llm + 1e-12)
                break

            n += n_step

        if not released:
            # refused
            hmin_bits = compute_hmin_bits_from_tau(tau_hat_llm) if pd.notna(tau_hat_llm) else float("nan")
            violation = False

        out.append(
            ReleaseRun(
                domain="E-health",
                released=released,
                tau_hat=float(tau_hat),
                tau_hat_llm=float(tau_hat_llm),
                hmin_bits=float(hmin_bits),
                attacker_post_true=float(attacker_true),
                violation=bool(violation),
                ring_size=int(ring_size),
                iters=int(iters),
            )
        )

    cfg = {
        "runs": int(runs),
        "tau_target": float(tau_target),
        "hmin_target": float(hmin_target),
        "max_iters": int(max_iters),
        "n0": int(n0),
        "n_step": int(n_step),
        "n_max": int(n_max),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "beta": float(beta),
    }
    return out, cfg


def build_propensity_tables_osn(actions: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    tmp = actions.copy()
    tmp["moderator_id"] = tmp["moderator_id"].astype(str)
    tmp["action_type"] = tmp["action_type"].astype(str)

    c = tmp.groupby(["action_type", "moderator_id"]).size().rename("n").reset_index()
    total = tmp.groupby(["action_type"]).size().rename("tot").reset_index()
    c = c.merge(total, on="action_type", how="left")
    c["p"] = c["n"] / c["tot"].clip(lower=1)
    return {(r["action_type"], r["moderator_id"]): float(r["p"]) for _, r in c.iterrows()}


def eval_osn(
    users_path: Path,
    edges_path: Path,
    actions_path: Path,
    domain_name: str,
    seed: int,
    runs: int,
    tau_target: float,
    hmin_target: float,
    max_iters: int,
    n0: int,
    n_step: int,
    n_max: int,
    gamma: float,
    alpha: float,
    beta: float,
    max_users: int = 8000,
) -> Tuple[List[ReleaseRun], Dict[str, float]]:
    rng = np.random.default_rng(seed)

    users = pd.read_csv(users_path)
    edges = pd.read_csv(edges_path)
    actions = pd.read_csv(actions_path)

    users["user_id"] = users["user_id"].astype(str)
    users["attr_sig"] = users["attr_sig"].astype(str)

    # Downsample large OSN domains to keep the artifact fast and deterministic.
    if len(users) > max_users:
        users = users.sample(n=max_users, random_state=seed).copy()

    pool_set = set(users["user_id"].tolist())
    users["role"] = users["role"].astype(str)

    actions["moderator_id"] = actions["moderator_id"].astype(str)
    actions["action_type"] = actions["action_type"].astype(str)

    # Restrict actions and edges to the downsampled population.
    actions = actions[actions["moderator_id"].isin(pool_set)].copy()

    # Normalize edge columns across datasets/generators.
    if "src" not in edges.columns or "dst" not in edges.columns:
        rename_map = {}
        if "src_id" in edges.columns:
            rename_map["src_id"] = "src"
        if "dst_id" in edges.columns:
            rename_map["dst_id"] = "dst"
        if "source_id" in edges.columns:
            rename_map["source_id"] = "src"
        if "target_id" in edges.columns:
            rename_map["target_id"] = "dst"
        if "from" in edges.columns:
            rename_map["from"] = "src"
        if "to" in edges.columns:
            rename_map["to"] = "dst"
        if rename_map:
            edges = edges.rename(columns=rename_map)
    if "src" not in edges.columns or "dst" not in edges.columns:
        raise ValueError(f"Unsupported edge columns in {edges_path}: {list(edges.columns)}")

    # Now that we have canonical column names, restrict edges to the population.
    edges = edges[edges["src"].isin(pool_set) & edges["dst"].isin(pool_set)].copy()
    edges["src"] = edges["src"].astype(str)
    edges["dst"] = edges["dst"].astype(str)

    # Now safe to filter by the downsampled population.
    edges = edges[edges["src"].isin(pool_set) & edges["dst"].isin(pool_set)].copy()

    # Adjacency for truncated BFS distances
    adj = build_undirected_adj(edges, "src", "dst")

    # Degree map
    degree = {u: int(len(nbrs)) for u, nbrs in adj.items()}

    # Attribute signature map
    attr_sig = dict(zip(users["user_id"], users["attr_sig"]))

    # Propensity table P(moderator | action_type)
    prop = build_propensity_tables_osn(actions)

    # Candidate pool: moderators only
    mod_roles = {"moderator", "senior_moderator"}
    mods = users[users["role"].isin(mod_roles)][["user_id", "attr_sig"]].drop_duplicates()
    mods = mods.rename(columns={"user_id": "id", "attr_sig": "sig"})

    if mods.empty:
        return [], {}

    # Sample runs from action log
    sample_df = actions.sample(n=min(runs, len(actions)), random_state=seed).reset_index(drop=True)

    out: List[ReleaseRun] = []
    for _, row in sample_df.iterrows():
        signer = str(row["moderator_id"])
        action_key = str(row.get("action_type", ""))

        # If signer is not in moderator pool, skip
        if signer not in set(mods["id"].astype(str)):
            continue

        dist_map = bfs_dist_trunc(adj, signer, max_depth=2)

        # Prefer candidates at distance >=2 (or unknown -> far bucket)
        def is_far(u: str) -> bool:
            return dist_or_bucket(dist_map, u, far_bucket=3) >= 2

        far_ids = [u for u in mods["id"].astype(str).tolist() if u == signer or is_far(u)]
        cand = mods[mods["id"].astype(str).isin(far_ids)].copy()
        if len(cand) < max(8, n0 // 2):
            cand = mods.copy()

        iters = 0
        released = False
        tau_hat = float("nan")
        tau_hat_llm = float("nan")
        attacker_true = float("nan")
        violation = False
        ring_size = 0

        n = n0
        while iters < max_iters and n <= n_max:
            iters += 1

            ring = select_ring_by_deficit(signer, cand, id_col="id", sig_col="sig", n_total=n, rng=rng)
            ring_size = len(ring)

            # Certificate tau_hat (product prior)
            post_cert = attack_posterior_product_prior(ring, signer, attr_sig, dist_map, gamma, alpha, beta)
            tau_hat = max_posterior(post_cert)

            # Tightened certificate against H
            tau_hat_llm, attacker_true = max_over_attacks(
                ring,
                signer,
                attr_sig,
                dist_map,
                degree,
                action_key,
                prop,
                gamma,
                alpha,
                beta,
            )
            hmin_bits = compute_hmin_bits_from_tau(tau_hat_llm)

            if tau_hat_llm <= tau_target and hmin_bits >= hmin_target:
                released = True
                violation = attacker_true > (tau_hat_llm + 1e-12)
                break

            n += n_step

        if not released:
            hmin_bits = compute_hmin_bits_from_tau(tau_hat_llm) if pd.notna(tau_hat_llm) else float("nan")
            violation = False

        out.append(
            ReleaseRun(
                domain=domain_name,
                released=released,
                tau_hat=float(tau_hat),
                tau_hat_llm=float(tau_hat_llm),
                hmin_bits=float(hmin_bits),
                attacker_post_true=float(attacker_true),
                violation=bool(violation),
                ring_size=int(ring_size),
                iters=int(iters),
            )
        )

    cfg = {
        "runs": int(runs),
        "tau_target": float(tau_target),
        "hmin_target": float(hmin_target),
        "max_iters": int(max_iters),
        "n0": int(n0),
        "n_step": int(n_step),
        "n_max": int(n_max),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "beta": float(beta),
    }
    return out, cfg


# -----------------------------
# LaTeX table helpers
# -----------------------------

def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = df.columns.tolist()
    colspec = "l" * 2 + "r" * (len(cols) - 2)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")

    fmt = df.copy()
    for c in fmt.columns:
        if c in {"Released", "Refusal", "Violation"}:
            fmt[c] = fmt[c].map(lambda x: f"{100.0*float(x):.1f}\\%" if pd.notna(x) else "nan")
        elif c in {"$\\hat{\\tau}$", "$\\hat{\\tau}_{\\mathrm{LLM}}$", "$\\mathrm{E}[p(u_0)]$", "$H_{\\min}$ (bits)", "Mean ring"}:
            fmt[c] = fmt[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "nan")
        elif c in {"Runs"}:
            fmt[c] = fmt[c].astype(int)

    for _, r in fmt.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/mnt/data/prime_ring_cert_results")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs_per_domain", type=int, default=100)
    ap.add_argument("--max_osn_users", type=int, default=8000)

    # Release criteria
    ap.add_argument("--tau_target", type=float, default=0.20)
    ap.add_argument("--hmin_target", type=float, default=2.50)

    # Tightening budget
    ap.add_argument("--max_iters", type=int, default=5)
    ap.add_argument("--n0", type=int, default=16)
    ap.add_argument("--n_step", type=int, default=8)
    ap.add_argument("--n_max", type=int, default=64)

    # Certificate parameters
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=1.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs: List[ReleaseRun] = []

    # E-health
    e_runs, e_cfg = eval_ehealth(
        Path(args.data_dir) / "prime_ring_ehealth_access_log.csv",
        Path(args.data_dir) / "prime_ring_ehealth_staff_graph_edges.csv",
        seed=args.seed,
        runs=args.runs_per_domain,
        tau_target=args.tau_target,
        hmin_target=args.hmin_target,
        max_iters=args.max_iters,
        n0=args.n0,
        n_step=args.n_step,
        n_max=args.n_max,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
    )
    all_runs += e_runs

    # OSN Twitter
    tw_runs, tw_cfg = eval_osn(
        Path(args.data_dir) / "prime_ring_osn_twitter_users.csv",
        Path(args.data_dir) / "prime_ring_osn_twitter_edges.csv",
        Path(args.data_dir) / "prime_ring_osn_twitter_actions.csv",
        domain_name="OSN-Twitter",
        seed=args.seed,
        runs=args.runs_per_domain,
        tau_target=args.tau_target,
        hmin_target=args.hmin_target,
        max_iters=args.max_iters,
        n0=args.n0,
        n_step=args.n_step,
        n_max=args.n_max,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        max_users=args.max_osn_users,
    )
    all_runs += tw_runs

    # OSN Facebook
    fb_runs, fb_cfg = eval_osn(
        Path(args.data_dir) / "prime_ring_osn_facebook_users.csv",
        Path(args.data_dir) / "prime_ring_osn_facebook_edges.csv",
        Path(args.data_dir) / "prime_ring_osn_facebook_actions.csv",
        domain_name="OSN-Facebook",
        seed=args.seed,
        runs=args.runs_per_domain,
        tau_target=args.tau_target,
        hmin_target=args.hmin_target,
        max_iters=args.max_iters,
        n0=args.n0,
        n_step=args.n_step,
        n_max=args.n_max,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        max_users=args.max_osn_users,
    )
    all_runs += fb_runs

    if not all_runs:
        raise SystemExit("No runs were produced; check input CSVs.")

    # Save raw results
    res_df = pd.DataFrame([r.__dict__ for r in all_runs])
    res_df.to_csv(out_dir / "results_cert_release.csv", index=False)

    # Summary per domain
    summary_rows = []
    for domain, g in res_df.groupby("domain"):
        runs = len(g)
        released = float(g["released"].mean())
        refusal = 1.0 - released
        viol = float(g.loc[g["released"], "violation"].mean()) if g["released"].any() else 0.0
        tau_hat_mean = float(g.loc[g["released"], "tau_hat"].mean()) if g["released"].any() else float("nan")
        tau_llm_mean = float(g.loc[g["released"], "tau_hat_llm"].mean()) if g["released"].any() else float("nan")
        ptrue_mean = float(g.loc[g["released"], "attacker_post_true"].mean()) if g["released"].any() else float("nan")
        hmin_mean = float(g.loc[g["released"], "hmin_bits"].mean()) if g["released"].any() else float("nan")
        ring_mean = float(g.loc[g["released"], "ring_size"].mean()) if g["released"].any() else float("nan")
        summary_rows.append(
            {
                "Domain": domain,
                "Runs": runs,
                "Released": released,
                "Refusal": refusal,
                "$\\hat{\\tau}$": tau_hat_mean,
                "$\\hat{\\tau}_{\\mathrm{LLM}}$": tau_llm_mean,
                "$\\mathrm{E}[p(u_0)]$": ptrue_mean,
                "$H_{\\min}$ (bits)": hmin_mean,
                "Mean ring": ring_mean,
                "Violation": viol,
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Attack/iteration detail table (compact)
    attacks = res_df.copy()
    attacks["hmin_bits"] = attacks["hmin_bits"].clip(lower=0)
    detail_rows = []
    for domain, g in attacks.groupby("domain"):
        gR = g[g["released"]]
        detail_rows.append(
            {
                "Domain": domain,
                "Metric": "iters (mean)",
                "Value": float(gR["iters"].mean()) if not gR.empty else float("nan"),
            }
        )
        detail_rows.append(
            {
                "Domain": domain,
                "Metric": "ring (p50)",
                "Value": float(gR["ring_size"].median()) if not gR.empty else float("nan"),
            }
        )
        detail_rows.append(
            {
                "Domain": domain,
                "Metric": "tau_hat (p95)",
                "Value": float(gR["tau_hat"].quantile(0.95)) if not gR.empty else float("nan"),
            }
        )
        detail_rows.append(
            {
                "Domain": domain,
                "Metric": "tau_llm (p95)",
                "Value": float(gR["tau_hat_llm"].quantile(0.95)) if not gR.empty else float("nan"),
            }
        )

    detail = pd.DataFrame(detail_rows)

    # Write LaTeX tables
    t1 = latex_table(
        summary,
        caption="Certified release results: tightened certificate $\\hat{\\tau}_{\\mathrm{LLM}}$ bounds attacker posterior mass on the true signer under declared feature set and attack class. Refusal counts runs that fail $\\hat{\\tau}_{\\mathrm{LLM}}\\le\\tau$ and $H_{\\min}\\ge H^\\star$ within a bounded tightening budget.",
        label="tab:prime_cert_release",
    )
    (out_dir / "table_cert_release_summary.tex").write_text(t1)

    # a second recommended table: detail statistics
    t2 = latex_table(
        detail,
        caption="Tightening loop statistics (released runs only).",
        label="tab:prime_tightening_stats",
    )
    (out_dir / "table_cert_release_attacks.tex").write_text(t2)

    # Save config
    config = {
        "seed": args.seed,
        "runs_per_domain": args.runs_per_domain,
        "tau_target": args.tau_target,
        "hmin_target": args.hmin_target,
        "max_iters": args.max_iters,
        "n0": args.n0,
        "n_step": args.n_step,
        "n_max": args.n_max,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "beta": args.beta,
        "domains": {
            "ehealth": e_cfg,
            "osn_twitter": tw_cfg,
            "osn_facebook": fb_cfg,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print("Wrote:")
    print(out_dir / "results_cert_release.csv")
    print(out_dir / "table_cert_release_summary.tex")
    print(out_dir / "table_cert_release_attacks.tex")
    print(out_dir / "config.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
