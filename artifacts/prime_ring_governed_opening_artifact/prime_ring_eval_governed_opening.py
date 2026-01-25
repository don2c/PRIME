#!/usr/bin/env python3
"""PRIME-Ring evaluation: governed opening and auditability.

This harness evaluates PRIME-Ring's accountability property on the synthetic
transcripts generated from the e-health and OSN datasets in this workspace.

We model the governed opening interface as:
  - Open(okey, transcript) -> signer_id if transcript is valid
  - Unauthorized parties without okey should fail
  - Corrupted transcripts should fail even for the auditor

We also log successful opens into an audit log and report completeness.

Inputs (expected to exist):
  - /mnt/data/prime_ring_ehealth_access_log.csv
  - /mnt/data/prime_ring_osn_twitter_actions.csv
  - /mnt/data/prime_ring_osn_twitter_users.csv
  - /mnt/data/prime_ring_osn_facebook_actions.csv
  - /mnt/data/prime_ring_osn_facebook_users.csv

Outputs (under --out_dir):
  - results.csv
  - audit_log.csv
  - transcripts_sample.csv
  - table_governed_opening.tex

Run:
  python prime_ring_eval_governed_opening.py --out_dir prime_ring_governed_opening_results --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes([x ^ y for x, y in zip(a, b)])


@dataclass
class OpenPayload:
    nonce_hex: str
    ct_hex: str
    tag_hex: str


def seal_signer(okey: bytes, signer: str, header: str, nonce: bytes) -> OpenPayload:
    signer_b = signer.encode("utf-8")
    ks = sha256(okey + nonce)
    ks = (ks * ((len(signer_b) // len(ks)) + 1))[: len(signer_b)]
    ct = xor_bytes(signer_b, ks)

    mac = hmac_sha256(okey, nonce + ct + header.encode("utf-8"))
    return OpenPayload(nonce.hex(), ct.hex(), mac.hex())


def open_transcript(okey: bytes, payload: OpenPayload, header: str) -> str | None:
    nonce = bytes.fromhex(payload.nonce_hex)
    ct = bytes.fromhex(payload.ct_hex)
    tag = bytes.fromhex(payload.tag_hex)

    exp = hmac_sha256(okey, nonce + ct + header.encode("utf-8"))
    if not hmac.compare_digest(tag, exp):
        return None

    ks = sha256(okey + nonce)
    ks = (ks * ((len(ct) // len(ks)) + 1))[: len(ct)]
    signer_b = xor_bytes(ct, ks)
    try:
        return signer_b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def sample_df(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_rows, replace=False)
    return df.iloc[idx].copy()


def make_transcripts_ehealth(df: pd.DataFrame, okey: bytes, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["signer_id"] = df["staff_id"].astype(str)
    df["header"] = (
        df["event_id"].astype(str)
        + "|" + df["epoch_t"].astype(str)
        + "|" + df["status_id"].astype(str)
        + "|" + df["plan_id"].astype(str)
    )

    rng = np.random.default_rng(seed)
    nonces = rng.integers(0, 256, size=(len(df), 16), dtype=np.uint8)

    payloads = [
        seal_signer(okey, s, h, bytes(n))
        for s, h, n in zip(df["signer_id"].tolist(), df["header"].tolist(), nonces)
    ]
    df["nonce_hex"] = [p.nonce_hex for p in payloads]
    df["ct_hex"] = [p.ct_hex for p in payloads]
    df["tag_hex"] = [p.tag_hex for p in payloads]

    df["domain"] = "E-health"
    df["transcript_id"] = df["event_id"].astype(str)
    return df[["domain", "transcript_id", "signer_id", "header", "nonce_hex", "ct_hex", "tag_hex"]]


def make_transcripts_osn(actions: pd.DataFrame, users: pd.DataFrame, okey: bytes, domain: str, seed: int) -> pd.DataFrame:
    acts = actions.copy()
    users = users.rename(columns={"user_id": "moderator_id"}).copy()

    df = acts.merge(users[["moderator_id"]], on="moderator_id", how="left")
    df["signer_id"] = df["moderator_id"].astype(str)
    df["header"] = (
        df["action_id"].astype(str)
        + "|" + df["epoch"].astype(str)
        + "|" + df["statusID"].astype(str)
        + "|" + df["planID"].astype(str)
    )

    rng = np.random.default_rng(seed)
    nonces = rng.integers(0, 256, size=(len(df), 16), dtype=np.uint8)

    payloads = [
        seal_signer(okey, s, h, bytes(n))
        for s, h, n in zip(df["signer_id"].tolist(), df["header"].tolist(), nonces)
    ]
    df["nonce_hex"] = [p.nonce_hex for p in payloads]
    df["ct_hex"] = [p.ct_hex for p in payloads]
    df["tag_hex"] = [p.tag_hex for p in payloads]

    df["domain"] = domain
    df["transcript_id"] = df["action_id"].astype(str)
    return df[["domain", "transcript_id", "signer_id", "header", "nonce_hex", "ct_hex", "tag_hex"]]


def pick_open_requests(df: pd.DataFrame, seed: int, open_frac: float, override_col: str | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)

    if override_col is not None and override_col in df.columns:
        mask = df[override_col].astype(int) == 1
        if mask.mean() >= 0.01:
            return mask

    return rng.random(len(df)) < open_frac


def corrupt_payload(row: pd.Series) -> OpenPayload:
    tag = bytearray.fromhex(row["tag_hex"])
    tag[0] ^= 0x01
    return OpenPayload(row["nonce_hex"], row["ct_hex"], bytes(tag).hex())


def eval_domain(transcripts: pd.DataFrame, domain_okey: bytes, seed: int, open_frac: float = 0.10, corrupt_frac: float = 0.05) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    unauthorized_key = sha256(b"unauthorized" + domain_okey)

    open_mask = pick_open_requests(transcripts, seed=seed, open_frac=open_frac)
    open_df = transcripts.loc[open_mask].copy()

    if len(open_df) == 0:
        return (
            {
                "Domain": transcripts["domain"].iloc[0],
                "N": len(transcripts),
                "Open N": 0,
                "Auditor SR": float("nan"),
                "Unauth SR": float("nan"),
                "False opens": float("nan"),
                "Corrupt accept": float("nan"),
                "Audit comp.": float("nan"),
                "Audit dupes": float("nan"),
            },
            pd.DataFrame(),
            transcripts.head(0),
        )

    corrupt_mask = rng.random(len(open_df)) < corrupt_frac
    valid_df = open_df.loc[~corrupt_mask].copy()
    corrupt_df = open_df.loc[corrupt_mask].copy()

    audit_rows = []
    correct = 0
    wrong = 0
    unauth_success = 0

    for _, r in valid_df.iterrows():
        payload = OpenPayload(r["nonce_hex"], r["ct_hex"], r["tag_hex"])
        got = open_transcript(domain_okey, payload, r["header"])
        if got is None:
            continue
        if got == r["signer_id"]:
            correct += 1
        else:
            wrong += 1

        audit_rows.append(
            {
                "domain": r["domain"],
                "transcript_id": r["transcript_id"],
                "opened_signer": got,
                "auditor_id": "auditor",
                "header": r["header"],
                "seed": seed,
            }
        )

        unauth = open_transcript(unauthorized_key, payload, r["header"])
        if unauth is not None:
            unauth_success += 1

    corrupt_accept = 0
    for _, r in corrupt_df.iterrows():
        payload = corrupt_payload(r)
        got = open_transcript(domain_okey, payload, r["header"])
        if got is not None:
            corrupt_accept += 1

        unauth = open_transcript(unauthorized_key, payload, r["header"])
        if unauth is not None:
            unauth_success += 1

    valid_n = len(valid_df)
    corrupt_n = len(corrupt_df)

    auditor_sr = correct / valid_n if valid_n else float("nan")
    unauth_sr = unauth_success / (len(open_df) + len(open_df)) if len(open_df) else float("nan")

    false_opens_rate = wrong / max(1, correct + wrong)
    corrupt_accept_rate = corrupt_accept / corrupt_n if corrupt_n else 0.0

    audit = pd.DataFrame(audit_rows)
    if len(audit) == 0:
        audit_comp = 1.0
        audit_dupes = 0.0
    else:
        expected = set(valid_df["transcript_id"].tolist())
        opened = set(audit["transcript_id"].tolist())
        audit_comp = len(opened) / max(1, len(expected))
        audit_dupes = float(audit.duplicated(subset=["domain", "transcript_id"]).sum())

    summary = {
        "Domain": transcripts["domain"].iloc[0],
        "N": len(transcripts),
        "Open N": len(open_df),
        "Auditor SR": auditor_sr,
        "Unauth SR": unauth_success / max(1, (len(open_df) * 2)),
        "False opens": false_opens_rate,
        "Corrupt accept": corrupt_accept_rate,
        "Audit comp.": audit_comp,
        "Audit dupes": audit_dupes,
    }

    sample_cols = ["domain", "transcript_id", "signer_id", "header", "nonce_hex", "ct_hex", "tag_hex"]
    sample = transcripts[sample_cols].sample(n=min(2000, len(transcripts)), random_state=seed)
    return summary, audit, sample


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = df.columns.tolist()
    colspec = "l" * 3 + "r" * (len(cols) - 3)
    bs = chr(92)
    nl = bs + bs

    lines = []
    lines.append(bs + "begin{table}[t]")
    lines.append(bs + "centering")
    lines.append(bs + f"caption{{{caption}}}")
    lines.append(bs + f"label{{{label}}}")
    lines.append(bs + f"begin{{tabular}}{{{colspec}}}")
    lines.append(bs + "toprule")
    lines.append(" & ".join(cols) + f" {nl}")
    lines.append(bs + "midrule")

    fmt = df.copy()
    for c in fmt.columns:
        if c in {"Auditor SR", "Unauth SR", "False opens", "Corrupt accept", "Audit comp."}:
            fmt[c] = fmt[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")

    for _, r in fmt.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + f" {nl}")

    lines.append(bs + "bottomrule")
    lines.append(bs + "end{tabular}")
    lines.append(bs + "end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/mnt/data/prime_ring_governed_opening_results")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=50000)
    ap.add_argument("--open_frac", type=float, default=0.10)
    ap.add_argument("--corrupt_frac", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Domain keys (fixed given seed for reproducibility)
    master = sha256(f"master|{args.seed}".encode("utf-8"))
    okey_e = sha256(master + b"ehealth")
    okey_tw = sha256(master + b"twitter")
    okey_fb = sha256(master + b"facebook")

    ehealth_path = Path(args.data_dir) / "prime_ring_ehealth_access_log.csv"
    tw_actions = Path(args.data_dir) / "prime_ring_osn_twitter_actions.csv"
    tw_users = Path(args.data_dir) / "prime_ring_osn_twitter_users.csv"
    fb_actions = Path(args.data_dir) / "prime_ring_osn_facebook_actions.csv"
    fb_users = Path(args.data_dir) / "prime_ring_osn_facebook_users.csv"

    e = sample_df(pd.read_csv(ehealth_path), max_rows=args.max_rows, seed=args.seed)
    tw_a = sample_df(pd.read_csv(tw_actions), max_rows=args.max_rows, seed=args.seed + 1)
    tw_u = pd.read_csv(tw_users)
    fb_a = sample_df(pd.read_csv(fb_actions), max_rows=args.max_rows, seed=args.seed + 2)
    fb_u = pd.read_csv(fb_users)

    te = make_transcripts_ehealth(e, okey_e, seed=args.seed)
    ttw = make_transcripts_osn(tw_a, tw_u, okey_tw, domain="OSN-Twitter", seed=args.seed + 3)
    tfb = make_transcripts_osn(fb_a, fb_u, okey_fb, domain="OSN-Facebook", seed=args.seed + 4)

    summaries = []
    audits = []
    samples = []

    for tdf, key in [(te, okey_e), (ttw, okey_tw), (tfb, okey_fb)]:
        s, audit, sample = eval_domain(tdf, key, seed=args.seed, open_frac=args.open_frac, corrupt_frac=args.corrupt_frac)
        summaries.append(s)
        if len(audit):
            audits.append(audit)
        samples.append(sample)

    res = pd.DataFrame(summaries)
    audit_all = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(columns=["domain", "transcript_id", "opened_signer", "auditor_id", "header", "seed"])
    sample_all = pd.concat(samples, ignore_index=True)

    res.to_csv(out_dir / "results.csv", index=False)
    audit_all.to_csv(out_dir / "audit_log.csv", index=False)
    sample_all.to_csv(out_dir / "transcripts_sample.csv", index=False)

    caption = "Governed opening correctness and auditability on PRIME-Ring transcripts. Unauthorized parties use a wrong opening key; corrupted transcripts flip one bit in the opening tag."
    tex = latex_table(res[["Domain", "N", "Open N", "Auditor SR", "Unauth SR", "False opens", "Corrupt accept", "Audit comp.", "Audit dupes"]], caption=caption, label="tab:governed_opening")
    (out_dir / "table_governed_opening.tex").write_text(tex)

    print("Wrote:")
    for fn in ["results.csv", "audit_log.csv", "transcripts_sample.csv", "table_governed_opening.tex"]:
        print(out_dir / fn)


if __name__ == "__main__":
    main()
