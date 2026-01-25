from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .datasets import DatasetSpec, load_users, load_ehealth_users
from .shape_sim import Plan, hmin_from_counts, top_k_signatures, simulate_shape


def _sample_signer_sig(avail: Dict[str, int], rng) -> str:
    keys = [k for k, v in avail.items() if v > 0]
    if not keys:
        raise ValueError("no available signatures")
    w = np.array([avail[k] for k in keys], dtype=float)
    w = w / w.sum()
    return str(rng.choice(keys, p=w))


def synthesize_ring_from_availability(
    baseline_sigs: List[str],
    avail: Dict[str, int],
    rng,
    nmin: int,
    H_star: float,
    delta_n_max: int,
    k_support: int,
) -> Tuple[int, int, int, float]:
    """Count-only ring synthesis + PadShape.

    - baseline_sigs defines the top-k support S0 used for the uniform exposure q*.
    - avail is the churned availability map (signature -> available keys).

    Returns (n_real, n_covers, n_total, Hmin).
    """

    total_avail = sum(max(0, v) for v in avail.values())
    if total_avail < nmin:
        return 0, 0, 0, 0.0

    # S0 and uniform q*
    S0 = top_k_signatures(baseline_sigs, k_support)
    k = len(S0)

    # pick signer signature and consume one key
    signer_sig = _sample_signer_sig(avail, rng)
    avail = dict(avail)
    avail[signer_sig] = max(0, avail.get(signer_sig, 0) - 1)

    # desired counts for uniform q* over S0
    base = nmin // k
    rem = nmin % k
    desired: Dict[str, int] = {s: base for s in S0}
    for s in S0[:rem]:
        desired[s] += 1

    counts: Dict[str, int] = {signer_sig: 1}

    # allocate towards S0 deficits using availability
    for s in S0:
        need = max(0, desired[s] - counts.get(s, 0))
        if need <= 0:
            continue
        take = min(need, avail.get(s, 0))
        if take > 0:
            counts[s] = counts.get(s, 0) + take
            avail[s] = avail.get(s, 0) - take

    # fill any remaining slots with whatever is available
    have = sum(counts.values())
    remaining = nmin - have
    if remaining > 0:
        # signatures sorted by remaining availability
        for s, v in sorted(avail.items(), key=lambda kv: (-kv[1], kv[0])):
            if remaining <= 0:
                break
            if v <= 0:
                continue
            take = min(remaining, v)
            counts[s] = counts.get(s, 0) + take
            avail[s] = v - take
            remaining -= take

    if remaining > 0:
        return 0, 0, 0, 0.0

    n_real = nmin

    # PadShape: raise Hmin on S0 by equalizing counts over S0
    S0_set = set(S0)
    n_covers = 0
    H = hmin_from_counts({s: counts.get(s, 0) for s in S0_set})
    while H < H_star and n_covers < delta_n_max:
        bucket = min(S0, key=lambda s: (counts.get(s, 0), s))
        counts[bucket] = counts.get(bucket, 0) + 1
        n_covers += 1
        H = hmin_from_counts({s: counts.get(s, 0) for s in S0_set})

    n_total = n_real + n_covers
    return n_real, n_covers, n_total, H


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, churn), g in df.groupby(["dataset", "churn"]):
        total = len(g)
        accepted = int(g["accepted"].sum())
        rej = total - accepted
        rej_len = int((g["reject_reason"] == "length").sum())
        rej_time = int((g["reject_reason"] == "time").sum())
        rej_pool = int((g["reject_reason"] == "pool").sum())

        g_acc = g[g["accepted"] == 1]
        if len(g_acc) == 0:
            out.append({
                "dataset": ds,
                "churn": churn,
                "runs": total,
                "accept_rate": 0.0,
                "reject_rate": 1.0,
                "reject_pool": rej_pool / total,
                "reject_length": rej_len / total,
                "reject_time": rej_time / total,
                "mean_n_total": float("nan"),
                "mean_covers": float("nan"),
                "mean_Hmin": float("nan"),
                "mean_pad_bytes": float("nan"),
                "mean_drift_ms": float("nan"),
                "p95_drift_ms": float("nan"),
            })
            continue

        out.append({
            "dataset": ds,
            "churn": churn,
            "runs": total,
            "accept_rate": accepted / total,
            "reject_rate": rej / total,
            "reject_pool": rej_pool / total,
            "reject_length": rej_len / total,
            "reject_time": rej_time / total,
            "mean_n_total": g_acc["n_total"].mean(),
            "mean_covers": g_acc["n_covers"].mean(),
            "mean_Hmin": g_acc["Hmin"].mean(),
            "mean_pad_bytes": g_acc["pad_bytes"].mean(),
            "mean_drift_ms": g_acc["drift_ms"].mean(),
            "p95_drift_ms": g_acc["drift_ms"].quantile(0.95),
        })

    return pd.DataFrame(out).sort_values(["dataset", "churn"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runs", type=int, default=400)

    # targets
    ap.add_argument("--tau", type=float, default=0.02, help="risk target; nmin=ceil(1/tau)")
    ap.add_argument("--Hstar", type=float, default=4.0, help="min-entropy target")
    ap.add_argument("--k-support", type=int, default=16, help="support size for uniform exposure")
    ap.add_argument("--delta-n-max", type=int, default=64)

    # plan envelope
    ap.add_argument("--L-bytes", type=int, default=4096)
    ap.add_argument("--Theta-ms", type=float, default=8.0)
    ap.add_argument("--delta-L", type=int, default=8)
    ap.add_argument("--delta-Theta-ms", type=float, default=0.2)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plan = Plan(
        plan_id=f"PLAN_L{args.L_bytes}_T{args.Theta_ms}ms",
        L_bytes=args.L_bytes,
        Theta_ms=args.Theta_ms,
        delta_L=args.delta_L,
        delta_Theta_ms=args.delta_Theta_ms,
    )

    # load datasets
    twitter = DatasetSpec(
        name="osn_twitter",
        users_path=args.data_dir / "prime_ring_osn_twitter_users.csv",
        id_col="user_id",
        sig_col="attr_sig",
    )
    facebook = DatasetSpec(
        name="osn_facebook",
        users_path=args.data_dir / "prime_ring_osn_facebook_users.csv",
        id_col="user_id",
        sig_col="attr_sig",
    )

    df_tw = load_users(twitter)
    df_fb = load_users(facebook)
    df_eh = load_ehealth_users(args.data_dir / "prime_ring_ehealth_access_log.csv")

    datasets = {
        "osn_twitter": df_tw.rename(columns={"attr_sig": "sig"})[["sig"]],
        "osn_facebook": df_fb.rename(columns={"attr_sig": "sig"})[["sig"]],
        "ehealth": df_eh.rename(columns={"attr_sig": "sig"})[["sig"]],
    }

    churn_levels = [0.0, 0.25, 0.50, 0.70]

    rng0 = np.random.default_rng(args.seed)
    nmin = int(np.ceil(1.0 / args.tau))

    records = []

    for ds_name, dfu in datasets.items():
        baseline_sigs = dfu["sig"].astype(str).tolist()
        base_counts = pd.Series(baseline_sigs).value_counts().to_dict()

        for churn in churn_levels:
            keep_p = 1.0 - churn
            for _ in range(args.runs):
                rrng = np.random.default_rng(rng0.integers(0, 2**32 - 1))

                # churn: thin each signature bucket independently
                avail = {s: int(rrng.binomial(int(n), keep_p)) for s, n in base_counts.items()}

                n_real, n_cov, n_tot, Hmin = synthesize_ring_from_availability(
                    baseline_sigs=baseline_sigs,
                    avail=avail,
                    rng=rrng,
                    nmin=nmin,
                    H_star=args.Hstar,
                    delta_n_max=args.delta_n_max,
                    k_support=args.k_support,
                )

                if n_tot == 0:
                    records.append({
                        "dataset": ds_name,
                        "churn": churn,
                        "accepted": 0,
                        "reject_reason": "pool",
                        "n_real": 0,
                        "n_covers": 0,
                        "n_total": 0,
                        "Hmin": 0.0,
                        "L_base": 0,
                        "pad_bytes": 0,
                        "L_final": 0,
                        "theta_base_ms": 0.0,
                        "theta_final_ms": 0.0,
                        "drift_ms": 0.0,
                    })
                    continue

                accepted, reason, L_base, pad_bytes, theta_base, theta_final, drift = simulate_shape(plan, n_tot, rrng)

                records.append({
                    "dataset": ds_name,
                    "churn": churn,
                    "accepted": 1 if accepted else 0,
                    "reject_reason": reason,
                    "n_real": n_real,
                    "n_covers": n_cov,
                    "n_total": n_tot,
                    "Hmin": Hmin,
                    "L_base": L_base,
                    "pad_bytes": pad_bytes,
                    "L_final": args.L_bytes if accepted else 0,
                    "theta_base_ms": theta_base,
                    "theta_final_ms": theta_final,
                    "drift_ms": drift,
                })

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_dir / "shape_runs.csv", index=False)

    summary = summarize(df)
    summary.to_csv(args.out_dir / "shape_summary.csv", index=False)

    meta = {
        "plan": {
            "plan_id": plan.plan_id,
            "L_bytes": plan.L_bytes,
            "Theta_ms": plan.Theta_ms,
            "delta_L": plan.delta_L,
            "delta_Theta_ms": plan.delta_Theta_ms,
        },
        "targets": {
            "tau": args.tau,
            "nmin": nmin,
            "Hstar": args.Hstar,
            "k_support": args.k_support,
            "delta_n_max": args.delta_n_max,
        },
        "runs": args.runs,
        "seed": args.seed,
        "churn_levels": churn_levels,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    md = []
    md.append("# Constant-shape enforcement results\n")
    md.append(
        f"Plan: `{plan.plan_id}` with L={plan.L_bytes} bytes, Θ={plan.Theta_ms} ms, δL={plan.delta_L} bytes, δΘ={plan.delta_Theta_ms} ms.\n"
    )
    md.append(f"Targets: τ={args.tau} (nmin={nmin}), H*={args.Hstar} bits, Δnmax={args.delta_n_max}.\n")
    md.append("\n## Summary (means computed over accepted runs)\n")
    md.append(summary.to_markdown(index=False))
    (args.out_dir / "SUMMARY.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
