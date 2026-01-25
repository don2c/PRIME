from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    users_path: Path
    id_col: str
    sig_col: str


def load_users(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.users_path)
    if spec.id_col not in df.columns:
        raise ValueError(f"{spec.name}: missing id_col={spec.id_col}")
    if spec.sig_col not in df.columns:
        raise ValueError(f"{spec.name}: missing sig_col={spec.sig_col}")
    df = df[[spec.id_col, spec.sig_col]].copy()
    df[spec.sig_col] = df[spec.sig_col].astype(str)
    return df


def load_ehealth_users(access_log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(access_log_path)
    need = {"staff_id", "attr_bucket"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"ehealth: missing columns: {sorted(missing)}")
    # unique staff with a stable attribute bucket (take mode, ties -> first)
    g = df.groupby("staff_id")["attr_bucket"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    out = g.reset_index().rename(columns={"attr_bucket": "attr_sig"})
    out["attr_sig"] = out["attr_sig"].astype(str)
    return out
