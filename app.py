# cspell:disable
"""
EuroLeague Pro DFS Suite — FanTeam 5+2 optimizer (7 players, PG/SG/SF/PF/C + 2 FLEX).

Cash and GPP portfolios from one click, ownership-aware, simulation-ranked.

Run:   streamlit run app.py
Needs: streamlit>=1.50, pandas, numpy, pulp, openpyxl (only for .xlsx uploads)
"""

import difflib
import hashlib
import html as _html
import io
import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
import streamlit as st

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
REQUIRED_POSITIONS = ["PG", "SG", "SF", "PF", "C"]
ROSTER_SIZE = 7
Z90 = 1.2815515655446004  # Floor = P10, Ceiling = P90

CALIBRATION_LOG = Path("calibration_log.csv")
OWNERSHIP_PARAMS_FILE = Path("ownership_params.json")
LOG_COLS = [
    "Slate",
    "Name",
    "TEAM",
    "Position",
    "Salary",
    "Proj",
    "Min",
    "Win_Prob",
    "Est_Own%",
    "Real_Own%",
]
DEFAULT_OWN_PARAMS = {"sharpness": 1.6, "w_val": 0.40}

TEAM_COLORS = {
    "OLY": ("#C8102E", "#FFFFFF"),
    "PAO": ("#007A33", "#FFFFFF"),
    "RMD": ("#FEBE10", "#0B0E14"),
    "RMB": ("#FEBE10", "#0B0E14"),
    "FBB": ("#002B5C", "#F6EB61"),
    "FBT": ("#002B5C", "#F6EB61"),
    "BAR": ("#A50044", "#004D98"),
    "MON": ("#E30613", "#FFFFFF"),
    "PAR": ("#222222", "#FFFFFF"),
    "VIR": ("#111111", "#FFFFFF"),
    "ASV": ("#111111", "#FFFFFF"),
    "PBB": ("#1D252D", "#00A3E0"),
    "HTA": ("#D00027", "#FFFFFF"),
    "BJK": ("#000000", "#FFFFFF"),
    "BAY": ("#DC052D", "#FFFFFF"),
    "ZAL": ("#006233", "#FFFFFF"),
    "EA7": ("#E30613", "#FFFFFF"),
    "MIL": ("#E4002B", "#FFFFFF"),
    "BKN": ("#002F6C", "#C8102E"),
    "MTA": ("#FCE300", "#003893"),
    "RED": ("#E30613", "#FFFFFF"),
    "ALB": ("#FFCD00", "#002B5C"),
    "VBC": ("#FF7A00", "#0B0E14"),
    "DUB": ("#B8860B", "#FFFFFF"),
}

# Strategy presets: everything in here can still be overridden under "Advanced".
PRESETS = {
    "Cash": {
        "Balanced": dict(
            blend=0.35, min_minutes=20.0, max_team=3, max_game=5, max_exp=100,
            blowout_pct=0.0,
        ),
        "Safe": dict(
            blend=0.50, min_minutes=22.0, max_team=3, max_game=4, max_exp=100,
            blowout_pct=5.0,
        ),
    },
    "GPP": {
        "Balanced": dict(
            blend=0.45, min_minutes=14.0, max_team=4, max_game=5, max_exp=65,
            blowout_pct=0.0, leverage=0.40, variety="Med", bring_back=True,
            env_boost=True,
        ),
        "Contrarian": dict(
            blend=0.55, min_minutes=12.0, max_team=4, max_game=6, max_exp=55,
            blowout_pct=0.0, leverage=0.75, variety="High", bring_back=True,
            env_boost=True,
        ),
    },
}
PRESET_HELP = {
    "Cash": {
        "Balanced": "35% floor blend, 20+ projected minutes, max 3 per team.",
        "Safe": "50% floor blend, 22+ minutes, 5% haircut on big-favourite starters.",
    },
    "GPP": {
        "Balanced": "45% ceiling blend, moderate leverage, bring-back on.",
        "Contrarian": "55% ceiling blend, heavy leverage, wider rotation of players.",
    },
}

VARIETY = {
    "Low": dict(randomness=0.03, team_share=0.30, min_unique=2),
    "Med": dict(randomness=0.06, team_share=0.45, min_unique=3),
    "High": dict(randomness=0.10, team_share=0.55, min_unique=3),
}

GPP_TARGETS = {
    "Large GPP (top 1%)": "Top1%",
    "Small GPP (top 10%)": "Top10%",
}
# metric column -> percentile of the simulated field that a lineup must beat
FIELD_PCTS = {"Cash_Top45%": 55.0, "Top10%": 90.0, "Top1%": 99.0}


def _flat(markup: str) -> str:
    """Collapses HTML to one line so Streamlit's markdown never sees an indented code block."""
    return re.sub(r"\s*\n\s*", " ", markup.strip())


def team_colors(code: str):
    if code in TEAM_COLORS:
        return TEAM_COLORS[code]
    hue = (sum(ord(c) for c in str(code)) * 37) % 360
    return (f"hsl({hue},45%,35%)", "#FFFFFF")


# ---------------------------------------------------------
# DATA: SAMPLE, NORMALISATION, VALIDATION
# ---------------------------------------------------------
def get_sample_data() -> pd.DataFrame:
    """Clearly labelled DEMO data (placeholder numbers) — upload a real slate."""
    csv_data = """Name,position,TEAM,Opp,Salary,Proj,Floor,Ceiling,Value,Min,Team_Pts,Pace,Win_Prob
TJ Shorts,PG,PBB,VIR,15.8,31.4,26.0,37.5,2.0,29.5,90.5,74.4,0.577
Nadir Hifi,SG,PBB,VIR,14.2,28.2,24.1,31.9,2.0,25.8,90.5,74.4,0.577
Tyson Ward,SF,PBB,VIR,10.5,20.8,16.2,26.4,1.98,26.5,90.5,74.4,0.577
Mikael Jantunen,PF,PBB,VIR,9.2,18.5,14.8,23.1,2.01,25.0,90.5,74.4,0.577
Kevarrius Hayes,C,PBB,VIR,8.4,16.9,13.0,21.5,2.01,22.0,90.5,74.4,0.577
Maodo Lo,PG,PBB,VIR,7.4,14.2,10.5,18.4,1.92,18.5,90.5,74.4,0.577
Collin Malcolm,SG,PBB,VIR,6.8,13.1,9.5,17.2,1.93,19.0,90.5,74.4,0.577
Tornike Shengelia,PF,VIR,PBB,13.8,26.0,21.5,31.0,1.9,27.5,87.0,74.4,0.423
Will Clyburn,SF,VIR,PBB,12.1,22.4,17.0,28.5,1.85,26.0,87.0,74.4,0.423
Isaia Cordinier,SG,VIR,PBB,11.4,21.8,16.8,27.4,1.91,25.5,87.0,74.4,0.423
Matt Morgan,PG,VIR,PBB,9.1,17.9,13.2,23.0,1.97,22.0,87.0,74.4,0.423
Ante Zizic,C,VIR,PBB,8.9,17.2,13.1,22.0,1.93,21.5,87.0,74.4,0.423
Alessandro Pajola,PG,VIR,PBB,7.3,14.0,10.4,18.2,1.92,21.0,87.0,74.4,0.423
Achille Polonara,PF,VIR,PBB,6.6,12.8,9.0,17.0,1.94,18.0,87.0,74.4,0.423
Elijah Bryant,SG,HTA,BJK,15.6,28.0,21.8,33.7,1.8,33.1,91.5,72.2,0.724
Jonathan Motley,C,HTA,BJK,13.2,25.6,19.8,31.4,1.94,26.4,91.5,72.2,0.724
Patrick Beverley,PG,HTA,BJK,11.0,21.5,16.5,27.2,1.95,25.5,91.5,72.2,0.724
Bruno Caboclo,PF,HTA,BJK,10.2,19.8,15.0,25.4,1.94,24.0,91.5,72.2,0.724
Ish Wainright,SF,HTA,BJK,7.8,15.1,11.4,19.2,1.94,23.0,91.5,72.2,0.724
Yam Madar,PG,HTA,BJK,7.5,14.6,10.8,19.0,1.95,19.5,91.5,72.2,0.724
Derek Needham,PG,BJK,HTA,10.4,19.9,15.2,25.1,1.91,27.0,79.5,72.2,0.276
Jonah Mathews,SG,BJK,HTA,10.8,20.5,15.5,26.0,1.90,28.0,79.5,72.2,0.276
Kelan Martin,SF,BJK,HTA,9.0,17.4,13.0,22.4,1.93,25.0,79.5,72.2,0.276
Dustin Sleva,PF,BJK,HTA,8.6,16.8,12.6,21.5,1.95,24.5,79.5,72.2,0.276
Emanuel Terry,C,BJK,HTA,8.2,16.1,12.0,20.8,1.96,23.5,79.5,72.2,0.276
Sasha Vezenkov,PF,OLY,FBB,16.5,33.0,27.5,39.2,2.0,31.0,86.0,70.1,0.640
Evan Fournier,SG,OLY,FBB,12.6,23.8,18.2,30.1,1.89,27.0,86.0,70.1,0.640
Kostas Papanikolaou,SF,OLY,FBB,9.4,18.1,14.0,22.8,1.93,25.5,86.0,70.1,0.640
Moses Wright,C,OLY,FBB,8.8,17.5,12.5,23.0,1.99,20.5,86.0,70.1,0.640
Thomas Walkup,PG,OLY,FBB,8.5,16.8,13.5,20.5,1.98,24.0,86.0,70.1,0.640
Alec Peters,PF,OLY,FBB,7.9,15.4,11.5,19.8,1.95,18.5,86.0,70.1,0.640
Nigel Hayes-Davis,SF,FBB,OLY,14.0,26.8,22.1,32.5,1.91,33.5,82.5,70.1,0.360
Wade Baldwin,PG,FBB,OLY,13.6,25.9,20.0,32.4,1.90,29.0,82.5,70.1,0.360
Bonzie Colson,PF,FBB,OLY,10.6,20.2,15.8,25.2,1.91,26.5,82.5,70.1,0.360
Marko Guduric,SG,FBB,OLY,9.8,18.7,14.2,23.8,1.91,24.0,82.5,70.1,0.360
Sertac Sanli,C,FBB,OLY,7.5,14.6,10.8,18.9,1.95,19.5,82.5,70.1,0.360
Nicolo Melli,C,FBB,OLY,7.2,14.1,10.5,18.2,1.96,20.0,82.5,70.1,0.360
"""
    return pd.read_csv(io.StringIO(csv_data.strip()))


def _percent_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False), errors="coerce"
    )


def normalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardises column names/types, drops unusable rows, derives game columns."""
    df = raw_df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    col_map = {
        "name": "Name",
        "player": "Name",
        "position": "Position",
        "pos": "Position",
        "team": "TEAM",
        "opp": "Opp",
        "opponent": "Opp",
        "salary": "Salary",
        "proj": "Proj",
        "floor": "Floor",
        "ceiling": "Ceiling",
        "value": "Value",
        "min": "Min",
        "minutes": "Min",
        "team_pts": "Team_Pts",
        "pace": "Pace",
        "win_prob": "Win_Prob",
        "own%": "Own%",
        "own": "Own%",
        "ownership": "Own%",
        "real_own%": "Real_Own%",
        "real_own": "Real_Own%",
        "max_exp%": "Max_Exp%",
    }
    df.rename(columns=lambda x: col_map.get(x.lower(), x), inplace=True)

    needed = ["Name", "Position", "TEAM", "Opp", "Salary", "Proj"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Projection file is missing required column(s): {', '.join(missing)}")

    numeric_cols = ["Salary", "Proj", "Floor", "Ceiling", "Value", "Min", "Team_Pts", "Pace", "Win_Prob"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Fallbacks so partial files still work
    zero_floor = df["Floor"] == 0
    df.loc[zero_floor & (df["Proj"] > 0), "Floor"] = (df["Proj"] * 0.65).round(1)
    zero_ceil = df["Ceiling"] == 0
    df.loc[zero_ceil & (df["Proj"] > 0), "Ceiling"] = (df["Proj"] * 1.35).round(1)

    df["Own%"] = _percent_series(df["Own%"]).fillna(0.0) if "Own%" in df.columns else 0.0
    if "Real_Own%" in df.columns:
        df["Real_Own%"] = _percent_series(df["Real_Own%"])
    if "Max_Exp%" in df.columns:
        df["Max_Exp%"] = _percent_series(df["Max_Exp%"]).clip(0.0, 100.0)
    else:
        df["Max_Exp%"] = np.nan  # blank = use the strategy default

    df["Name"] = df["Name"].astype(str).str.strip()
    df["Position"] = df["Position"].astype(str).str.upper().str.strip()
    df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip()
    df["Opp"] = df["Opp"].astype(str).str.upper().str.strip()

    bad = (
        (df["Salary"] <= 0)
        | (~df["Position"].isin(REQUIRED_POSITIONS))
        | (df["Name"].isin(["", "nan", "None"]))
    )
    dropped = int(bad.sum())
    df = df[~bad].reset_index(drop=True)

    dup = df.duplicated("Name", keep=False)
    if dup.any():
        df.loc[dup, "Name"] = df.loc[dup, "Name"] + " (" + df.loc[dup, "TEAM"] + ")"

    df["Game_Key"] = [" vs ".join(sorted([t, o])) for t, o in zip(df["TEAM"], df["Opp"])]
    team_pts_map = df.groupby("TEAM")["Team_Pts"].median().to_dict()
    df["Opp_Pts"] = df["Opp"].map(team_pts_map).fillna(df["Team_Pts"])
    df["Game_Total"] = (df["Team_Pts"] + df["Opp_Pts"]).round(1)
    df["Favored_By"] = (df["Team_Pts"] - df["Opp_Pts"]).round(1)
    df["Spread"] = (-df["Favored_By"]).round(1)
    df.attrs["dropped_rows"] = dropped
    return df


def validate_slate_data(df: pd.DataFrame) -> dict:
    cols = ["Name", "TEAM", "Position", "Floor", "Proj", "Ceiling"]
    floor_gt_proj = df[df["Floor"] > df["Proj"]][cols]
    ceil_lt_proj = df[df["Ceiling"] < df["Proj"]][cols]
    zero_floor = df[(df["Proj"] > 0) & (df["Floor"] <= 0)][cols]
    team_mins = df.groupby("TEAM")["Min"].sum().reset_index(name="Total_Team_Min").round(1)
    high_team_mins = team_mins[team_mins["Total_Team_Min"] > 200.0]
    return {
        "floor_gt_proj": floor_gt_proj,
        "ceil_lt_proj": ceil_lt_proj,
        "zero_floor": zero_floor,
        "high_team_mins": high_team_mins,
        "total_issues": len(floor_gt_proj) + len(ceil_lt_proj) + len(zero_floor) + len(high_team_mins),
    }


def repair_slate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df.copy()
    zero_mask = (fixed["Proj"] > 0) & (fixed["Floor"] <= 0)
    fixed.loc[zero_mask, "Floor"] = (fixed.loc[zero_mask, "Proj"] * 0.65).round(1)
    fixed["Floor"] = np.minimum(fixed["Floor"], fixed["Proj"])
    fixed["Ceiling"] = np.maximum(fixed["Ceiling"], fixed["Proj"])
    return fixed


# ---------------------------------------------------------
# OWNERSHIP: ESTIMATE, REAL (POST-LOCK) INPUT, CALIBRATION
# ---------------------------------------------------------
def ownership_weights(w_val: float) -> dict:
    """Value weight is fitted; the rest keep their relative proportions."""
    rest = max(0.0, 1.0 - w_val)
    return {
        "val": w_val,
        "proj": rest * 0.35 / 0.60,
        "min": rest * 0.15 / 0.60,
        "win": rest * 0.10 / 0.60,
    }


def estimate_ownership(df: pd.DataFrame, sharpness: float = 1.6, w_val: float = 0.40) -> pd.Series:
    """
    Softmax of z-scored Value, Proj, Min and Win_Prob, gated by minutes, then scaled so the
    slate sums to 700% (7 roster spots) with a 75% cap per player.
    """
    if len(df) == 0:
        return pd.Series(dtype=float)
    w = ownership_weights(w_val)

    def _z(s: pd.Series) -> pd.Series:
        sd = s.std()
        if sd == 0 or pd.isna(sd):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / sd

    value = df["Proj"] / df["Salary"].replace(0, np.nan).fillna(10.0)
    z_total = (
        w["val"] * _z(value)
        + w["proj"] * _z(df["Proj"])
        + w["min"] * _z(df["Min"])
        + w["win"] * _z(df["Win_Prob"])
    )
    logits = (z_total * sharpness).clip(-15.0, 15.0)
    weights = np.exp(logits - logits.max()).to_numpy()
    gate = 1.0 / (1.0 + np.exp(-(df["Min"].to_numpy(float) - 12.0) / 3.0))
    weights = weights * gate

    own = pd.Series(700.0 * (weights / (weights.sum() + 1e-9)), index=df.index)
    cap = 75.0
    for _ in range(10):
        over = own > cap
        if not over.any():
            break
        own[over] = cap
        remaining = 700.0 - float(own[over].sum())
        under = ~over
        under_sum = float(own[under].sum())
        if under.any() and under_sum > 0:
            own[under] = own[under] * (remaining / under_sum)
    return own.clip(0.1, cap).round(1)


def load_ownership_params() -> dict:
    try:
        if OWNERSHIP_PARAMS_FILE.exists():
            saved = json.loads(OWNERSHIP_PARAMS_FILE.read_text())
            return {
                "sharpness": float(saved.get("sharpness", DEFAULT_OWN_PARAMS["sharpness"])),
                "w_val": float(saved.get("w_val", DEFAULT_OWN_PARAMS["w_val"])),
            }
    except Exception:
        pass
    return dict(DEFAULT_OWN_PARAMS)


def save_ownership_params(params: dict) -> None:
    OWNERSHIP_PARAMS_FILE.write_text(json.dumps(params))


def parse_real_ownership(text: str, names) -> tuple:
    """
    Parses pasted `Name, 12.5%` lines (comma / tab / semicolon separated; header lines ignored).
    Returns (matched {name: own%}, unmatched [names]).
    """
    lookup = {str(n).lower().strip(): n for n in names}
    parsed = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"[,\t;]", line) if p.strip() != ""]
        if len(parts) < 2:
            continue
        try:
            val = float(parts[-1].replace("%", ""))
        except ValueError:
            continue  # header row
        parsed.append((" ".join(parts[:-1]).strip(), val))
    if parsed and max(v for _, v in parsed) <= 1.0:
        parsed = [(n, v * 100.0) for n, v in parsed]

    matched, unmatched = {}, []
    for name, val in parsed:
        key = name.lower()
        if key in lookup:
            matched[lookup[key]] = val
            continue
        close = difflib.get_close_matches(key, list(lookup.keys()), n=1, cutoff=0.85)
        if close:
            matched[lookup[close[0]]] = val
        else:
            unmatched.append(name)
    return matched, unmatched


def read_calibration_log() -> pd.DataFrame:
    try:
        if CALIBRATION_LOG.exists():
            return pd.read_csv(CALIBRATION_LOG)
    except Exception:
        pass
    return pd.DataFrame(columns=LOG_COLS)


def append_calibration_log(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Stores the whole slate (Real_Own% blank where unknown) so z-scores match live use."""
    rows = df.copy()
    rows["Slate"] = label
    rows["Real_Own%"] = rows["Real_Own%"] if "Real_Own%" in rows.columns else np.nan
    rows = rows[[c for c in LOG_COLS if c in rows.columns]]
    log = read_calibration_log()
    log = log[log["Slate"] != label]
    log = pd.concat([log, rows], ignore_index=True)
    log.to_csv(CALIBRATION_LOG, index=False)
    return log


def fit_ownership_params(log: pd.DataFrame) -> dict:
    """Grid-searches sharpness x value-weight to minimise MAE vs logged real ownership."""
    slates = [g.reset_index(drop=True) for _, g in log.groupby("Slate")]
    slates = [g for g in slates if g["Real_Own%"].notna().sum() >= 5]
    if not slates:
        return {}
    best = {"mae": 1e9}
    for sharp in np.round(np.linspace(0.5, 3.0, 26), 2):
        for w_val in (0.2, 0.3, 0.4, 0.5, 0.6):
            errs = []
            for g in slates:
                pred = estimate_ownership(g, sharpness=float(sharp), w_val=w_val).to_numpy()
                m = g["Real_Own%"].notna().to_numpy()
                errs.append(np.abs(pred[m] - g["Real_Own%"].to_numpy(float)[m]))
            mae = float(np.concatenate(errs).mean())
            if mae < best["mae"]:
                best = {"sharpness": float(sharp), "w_val": float(w_val), "mae": mae}
    return best


# ---------------------------------------------------------
# SCORING
# ---------------------------------------------------------
def leverage_lambda(leverage: float) -> float:
    """0-1 leverage slider -> points of Opt_Score removed per 1% of ownership."""
    return round(0.15 * leverage, 4)


def leverage_constraints(leverage: float) -> tuple:
    """0-1 leverage slider -> (max total lineup own %, chalk threshold %, max chalk players)."""
    if leverage <= 0:
        return 700.0, 30.0, ROSTER_SIZE
    return float(round(400 - 220 * leverage)), 30.0, int(round(4 - 2 * leverage))


def compute_metrics_after_edit(
    df: pd.DataFrame,
    kind: str,
    blend: float,
    env_boost: bool,
    blowout_threshold: float,
    blowout_pct: float,
    leverage_lam: float,
) -> pd.DataFrame:
    """
    Recomputes everything derived from the (possibly edited) Proj / Floor / Ceiling / Salary.
    Blowout haircut hits Proj, Floor and Ceiling together so the three stay consistent.
    Ownership is NOT touched here — it is estimated once on the full slate.
    """
    out = df.copy()
    haircut = np.where(
        (out["Favored_By"] > blowout_threshold) & (out["Min"] >= 24.0),
        1.0 - blowout_pct / 100.0,
        1.0,
    )
    out["Adj_Proj"] = (out["Proj"] * haircut).round(2)
    out["Adj_Floor"] = (out["Floor"] * haircut).round(2)
    out["Adj_Ceiling"] = (out["Ceiling"] * haircut).round(2)
    out["Value"] = (out["Adj_Proj"] / out["Salary"].replace(0, np.nan).fillna(1.0)).round(2)

    if kind == "Cash":
        score = (1.0 - blend) * out["Adj_Proj"] + blend * out["Adj_Floor"]
    else:
        score = (1.0 - blend) * out["Adj_Proj"] + blend * out["Adj_Ceiling"]
        if env_boost and out["Game_Total"].max() > 0:
            lo, hi = out["Game_Total"].min(), out["Game_Total"].max()
            env_norm = (out["Game_Total"] - lo) / (hi - lo + 1e-6)
            score = score * (1.0 + 0.035 * env_norm)
        score = score - leverage_lam * out["Own%"]
    out["Opt_Score"] = score.round(3)
    return out


def assign_roster_slots(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """Orders the 7 players into PG, SG, SF, PF, C, FLEX 1, FLEX 2."""
    df = lineup_df.sort_values(by=["Salary", "Proj"], ascending=[False, False]).copy()
    assigned, used = {}, set()
    for pos in REQUIRED_POSITIONS:
        cands = df[(df["Position"] == pos) & (~df.index.isin(used))]
        if not cands.empty:
            assigned[pos] = cands.index[0]
            used.add(cands.index[0])
    rows = []
    for pos in REQUIRED_POSITIONS:
        if pos in assigned:
            row = df.loc[assigned[pos]].copy()
            row["Slot"] = pos
            rows.append(row)
    for n, idx in enumerate([i for i in df.index if i not in used], start=1):
        row = df.loc[idx].copy()
        row["Slot"] = f"FLEX {n}"
        rows.append(row)
    result = pd.DataFrame(rows).reset_index(drop=True)
    return result[["Slot"] + [c for c in result.columns if c != "Slot"]]


# ---------------------------------------------------------
# SOLVER
# ---------------------------------------------------------
def _binary_var(name: str):
    try:
        return pulp.LpVariable(name, lowBound=0, upBound=1, cat="Binary")
    except TypeError:
        return pulp.LpVariable(name, 0, 1, pulp.LpBinary)


def diagnose_settings(data, budget, min_budget, locked, stacks, max_team, force_pairs, ban_pairs) -> list:
    """Plain-English reasons a build cannot work, checked before calling the solver."""
    msgs = []
    names = set(data["Name"])
    if len(data) < ROSTER_SIZE:
        msgs.append(f"Only {len(data)} players are in the pool; at least {ROSTER_SIZE} are needed.")
    for pos in REQUIRED_POSITIONS:
        if not (data["Position"] == pos).any():
            msgs.append(f"No {pos} in the pool. Lower the minimum minutes or re-include a team.")
    if min_budget > budget:
        msgs.append("Minimum salary spent is above the salary cap.")
    if len(data) >= ROSTER_SIZE and data["Salary"].nsmallest(ROSTER_SIZE).sum() > budget:
        msgs.append("Even the seven cheapest players exceed the salary cap.")
    for n in locked:
        if n not in names:
            msgs.append(f"Locked player '{n}' is not in the pool.")
    lk = data[data["Name"].isin(locked)]
    if len(lk) > ROSTER_SIZE:
        msgs.append("More than 7 locked players.")
    if len(lk) and lk["Salary"].sum() > budget:
        msgs.append("Locked players alone exceed the salary cap.")
    if len(lk) and lk["TEAM"].value_counts().max() > max_team:
        msgs.append(f"Locked players include more than {max_team} from one team.")
    for team, m in stacks.items():
        if m > max_team:
            msgs.append(f"Stack of {m} from {team} exceeds the max {max_team} per team.")
        if (data["TEAM"] == team).sum() < m:
            msgs.append(f"Not enough {team} players in the pool for a stack of {m}.")
    for a, b in ban_pairs:
        if a in locked and b in locked:
            msgs.append(f"{a} and {b} are both locked but banned together.")
    return msgs


def generate_lineups(
    data: pd.DataFrame,
    num_lineups: int,
    budget: float,
    min_budget: float,
    locked: list,
    stacks: dict,
    bring_back_enabled: bool,
    auto_bring_back_min: int,
    max_team: int,
    max_per_game: int,
    min_unique: int,
    randomness: float,
    team_shock_share: float,
    max_lineup_own: float,
    max_chalk_count: int,
    chalk_threshold: float,
    linked_pairs: list,
    ban_pairs: list,
    solver_time_limit: float = 5.0,
    seed: int = 42,
) -> list:
    """
    Builds the MILP once, then re-solves with a fresh objective for each lineup,
    adding uniqueness cuts and exposure bans as it goes.
    """
    if len(data) < ROSTER_SIZE:
        return []
    for pos in REQUIRED_POSITIONS:
        if pos not in data["Position"].unique():
            return []

    data = data.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    prob = pulp.LpProblem("FanTeam_EuroLeague_Optimizer", pulp.LpMaximize)
    x = {i: _binary_var(f"Player_{i}") for i in data.index}
    salary = data["Salary"].to_numpy(float)
    own = data["Own%"].to_numpy(float) if "Own%" in data.columns else np.zeros(len(data))

    prob += (pulp.lpSum([x[i] for i in data.index]) == ROSTER_SIZE, "Roster_Size")
    prob += (pulp.lpSum([float(salary[i]) * x[i] for i in data.index]) <= budget, "Max_Salary")
    prob += (pulp.lpSum([float(salary[i]) * x[i] for i in data.index]) >= min_budget, "Min_Salary")

    for pos in REQUIRED_POSITIONS:
        prob += (
            pulp.lpSum([x[i] for i in data.index if data.loc[i, "Position"] == pos]) >= 1,
            f"Min_Pos_{pos}",
        )
    for team in data["TEAM"].unique():
        prob += (
            pulp.lpSum([x[i] for i in data.index if data.loc[i, "TEAM"] == team]) <= max_team,
            f"Max_Team_{team}",
        )
    if max_per_game < ROSTER_SIZE:
        for g in data["Game_Key"].unique():
            prob += (
                pulp.lpSum([x[i] for i in data.index if data.loc[i, "Game_Key"] == g]) <= max_per_game,
                "Max_Game_" + re.sub(r"\W", "_", g),
            )

    for name in locked:
        for idx in data[data["Name"] == name].index:
            prob += (x[idx] == 1, f"Lock_{idx}")

    for team, min_count in stacks.items():
        prob += (
            pulp.lpSum([x[i] for i in data.index if data.loc[i, "TEAM"] == team]) >= min_count,
            f"Stack_{team}",
        )
        if bring_back_enabled:
            opp_teams = data.loc[data["TEAM"] == team, "Opp"].unique().tolist()
            opp_idx = data[data["TEAM"].isin(opp_teams)].index.tolist()
            if opp_idx:
                prob += (pulp.lpSum([x[i] for i in opp_idx]) >= 1, f"BringBack_{team}")

    if bring_back_enabled and auto_bring_back_min >= 2:
        for team in data["TEAM"].unique():
            team_idx = data[data["TEAM"] == team].index.tolist()
            opp_teams = data.loc[data["TEAM"] == team, "Opp"].unique().tolist()
            opp_idx = data[data["TEAM"].isin(opp_teams)].index.tolist()
            if len(team_idx) >= auto_bring_back_min and opp_idx:
                ind = _binary_var(f"StackInd_{team}")
                prob += (
                    pulp.lpSum([x[i] for i in team_idx]) - (auto_bring_back_min - 1)
                    <= ROSTER_SIZE * ind,
                    f"AutoStack_{team}",
                )
                prob += (pulp.lpSum([x[i] for i in opp_idx]) >= ind, f"AutoBringBack_{team}")

    if max_lineup_own < 700.0:
        prob += (
            pulp.lpSum([float(own[i]) * x[i] for i in data.index]) <= max_lineup_own,
            "Max_Lineup_Own",
        )
    if max_chalk_count < ROSTER_SIZE:
        chalk = [i for i in data.index if own[i] >= chalk_threshold]
        if chalk:
            prob += (pulp.lpSum([x[i] for i in chalk]) <= max_chalk_count, "Max_Chalk")

    for k, (a, b) in enumerate(linked_pairs):  # both or neither
        ia, ib = data[data["Name"] == a].index.tolist(), data[data["Name"] == b].index.tolist()
        if ia and ib:
            prob += (x[ia[0]] - x[ib[0]] == 0, f"Linked_{k}")
    for k, (a, b) in enumerate(ban_pairs):  # never together
        ia, ib = data[data["Name"] == a].index.tolist(), data[data["Name"] == b].index.tolist()
        if ia and ib:
            prob += (x[ia[0]] + x[ib[0]] <= 1, f"Banned_{k}")

    max_counts = portfolio_max_counts(data, locked, num_lineups)
    counts = {i: 0 for i in data.index}
    banned = set()
    lineups = []

    try:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    except TypeError:
        solver = pulp.PULP_CBC_CMD(msg=False)

    base = data["Opt_Score"].to_numpy(float)
    teams = data["TEAM"].unique()
    games = data["Game_Key"].unique()

    for n in range(num_lineups):
        for i in data.index:
            if i not in banned and counts[i] >= max_counts[data.loc[i, "Name"]]:
                prob += (x[i] == 0, f"ExpBan_{i}_{n}")
                banned.add(i)

        if randomness > 0 and n > 0:
            t_sh = {t: rng.normal(0.0, randomness * team_shock_share) for t in teams}
            g_sh = {g: rng.normal(0.0, randomness * 0.25) for g in games}
            p_sh = rng.normal(0.0, randomness * max(0.1, 1.0 - team_shock_share * 0.5), size=len(data))
            mult = 1.0 + data["TEAM"].map(t_sh).to_numpy(float) + data["Game_Key"].map(g_sh).to_numpy(float) + p_sh
            current = base * np.clip(mult, 0.5, 1.6)
        else:
            current = base

        prob.setObjective(pulp.lpSum([float(current[i]) * x[i] for i in data.index]))
        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            break
        chosen = [i for i in data.index if x[i].varValue is not None and x[i].varValue > 0.5]
        if len(chosen) != ROSTER_SIZE:
            break
        for i in chosen:
            counts[i] += 1
        prob += (
            pulp.lpSum([x[i] for i in chosen]) <= ROSTER_SIZE - min_unique,
            f"Unique_{n}",
        )
        lineups.append(assign_roster_slots(data.loc[chosen].copy()))
    return lineups


def portfolio_max_counts(data: pd.DataFrame, locked, n: int) -> dict:
    """Per-player appearance cap for a portfolio of n lineups (locked players are exempt)."""
    caps = {}
    for _, r in data.iterrows():
        if r["Name"] in locked:
            caps[r["Name"]] = n
            continue
        pct = r["Max_Exp%"] if "Max_Exp%" in data.columns and not pd.isna(r["Max_Exp%"]) else 100.0
        if pct <= 0:
            caps[r["Name"]] = 0
        elif pct >= 100:
            caps[r["Name"]] = n
        else:
            caps[r["Name"]] = max(1, math.floor(pct / 100.0 * n + 1e-9))
    return caps


def select_diverse(lineups: list, order: list, n: int, min_unique: int, max_counts: dict) -> list:
    """
    Greedy pick of n lineups from a ranked candidate list, respecting a minimum number of
    different players between any two picks and per-player exposure caps. If the pool runs
    dry, the uniqueness requirement is relaxed one step at a time.
    """
    chosen, counts = [], {}
    name_sets = [set(l["Name"]) for l in lineups]
    for need in range(min_unique, 0, -1):
        for ci in order:
            if len(chosen) >= n:
                break
            if ci in chosen:
                continue
            if any(len(name_sets[ci] & name_sets[c]) > ROSTER_SIZE - need for c in chosen):
                continue
            if any(counts.get(nm, 0) >= max_counts.get(nm, n) for nm in name_sets[ci]):
                continue
            chosen.append(ci)
            for nm in name_sets[ci]:
                counts[nm] = counts.get(nm, 0) + 1
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------
# SIMULATION: CORRELATED PLAYERS + OWNERSHIP-WEIGHTED FIELD
# ---------------------------------------------------------
def simulate_player_scores(df: pd.DataFrame, num_sims: int, rho_game: float, rho_team: float, seed: int = 101) -> np.ndarray:
    """
    (players x sims) fantasy scores. Floor = P10, Ceiling = P90, Proj = median (split normal).
    Players share a game factor (pace / shootout) and a team factor, so teammates correlate.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    mu = df["Adj_Proj"].to_numpy(float)
    sd_dn = np.maximum(0.5, (mu - df["Adj_Floor"].to_numpy(float)) / Z90)
    sd_up = np.maximum(0.5, (df["Adj_Ceiling"].to_numpy(float) - mu) / Z90)
    g_code, _ = pd.factorize(df["Game_Key"])
    t_code, _ = pd.factorize(df["TEAM"])
    z_game = rng.standard_normal((int(g_code.max()) + 1, num_sims))
    z_team = rng.standard_normal((int(t_code.max()) + 1, num_sims))
    z_ind = rng.standard_normal((n, num_sims))
    rho_ind = max(0.05, 1.0 - rho_game - rho_team)
    z = math.sqrt(rho_game) * z_game[g_code] + math.sqrt(rho_team) * z_team[t_code] + math.sqrt(rho_ind) * z_ind
    sims = np.where(
        z < 0,
        np.maximum(0.0, mu[:, None] + z * sd_dn[:, None]),
        mu[:, None] + z * sd_up[:, None],
    )
    return sims.astype(np.float32)


def simulate_field_matrix(
    df: pd.DataFrame,
    budget: float,
    n_field: int = 4000,
    seed: int = 7,
    min_gap: float = 10.0,
    keep_frac: float = 0.15,
) -> np.ndarray:
    """
    (field lineups x players) 0/1 matrix. Lineups are sampled in proportion to ownership over the
    FULL slate (fading a player never removes them from the field), kept if valid (7 players, all
    five positions, under the cap, spending within `min_gap` of it), and then only the top
    `keep_frac` by projection survive. Real entrants are sharper than random picks, so without that
    last step the field would be unrealistically weak and every lineup would look unbeatable.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    own = np.maximum(df["Own%"].to_numpy(float), 0.05)
    logp = np.log(own / own.sum())
    sal = df["Salary"].to_numpy(float)
    proj = df["Adj_Proj"].to_numpy(float)
    pos = df["Position"].to_numpy()
    onehot = np.stack([(pos == p) for p in REQUIRED_POSITIONS], axis=1).astype(np.int8)
    if (onehot.sum(0) == 0).any():
        return np.zeros((0, n), dtype=np.float32)

    keep_frac = float(min(1.0, max(0.05, keep_frac)))
    need = int(n_field / keep_frac)
    kept, total = [], 0
    for gap in (min_gap, min_gap * 2, 1e9):
        kept, total = [], 0
        for _ in range(40):
            keys = logp[None, :] + rng.gumbel(size=(20000, n))
            idx = np.argpartition(-keys, ROSTER_SIZE - 1, axis=1)[:, :ROSTER_SIZE]
            spend = sal[idx].sum(1)
            ok = (spend <= budget) & (spend >= budget - gap) & (onehot[idx].sum(1) >= 1).all(1)
            if ok.any():
                kept.append(idx[ok])
                total += int(ok.sum())
            if total >= need:
                break
        if total >= min(need, 2000):
            break
    if total == 0:
        return np.zeros((0, n), dtype=np.float32)
    sel = np.vstack(kept)[:need]
    if keep_frac < 1.0 and len(sel) > 200:
        order = np.argsort(-proj[sel].sum(1))
        sel = sel[order[: max(200, int(len(sel) * keep_frac))]]
    m = np.zeros((len(sel), n), dtype=np.float32)
    m[np.arange(len(sel))[:, None], sel] = 1.0
    return m


def field_thresholds(field_matrix: np.ndarray, sims: np.ndarray):
    """Per-simulation score a lineup must reach to beat the chosen share of the field."""
    if field_matrix.shape[0] < 50:
        return None
    field_scores = field_matrix @ sims
    qs = list(FIELD_PCTS.values())
    return dict(zip(FIELD_PCTS.keys(), np.percentile(field_scores, qs, axis=0)))


def evaluate_lineups(lineups: list, df: pd.DataFrame, sims: np.ndarray, thresholds) -> pd.DataFrame:
    name_to_idx = {n: i for i, n in enumerate(df["Name"])}
    m = np.zeros((len(lineups), len(df)), dtype=np.float32)
    for li, l in enumerate(lineups):
        for nm in l["Name"]:
            m[li, name_to_idx[nm]] = 1.0
    scores = m @ sims
    rec = pd.DataFrame(
        {
            "Lineup_#": np.arange(1, len(lineups) + 1),
            "Sim_Mean": scores.mean(1),
            "Sim_P10": np.percentile(scores, 10, axis=1),
            "Sim_P90": np.percentile(scores, 90, axis=1),
        }
    )
    for col in FIELD_PCTS:
        rec[col] = (scores >= thresholds[col][None, :]).mean(1) * 100.0 if thresholds else np.nan
    return rec.round(1)


def build_compact_export(lineups: list, metrics: pd.DataFrame, budget: float) -> pd.DataFrame:
    """One row per lineup in PG, SG, SF, PF, C, FLEX 1, FLEX 2 order for quick entry."""
    rows = []
    for idx, l in enumerate(lineups):
        row = {"Lineup_#": idx + 1}
        for _, p in l.iterrows():
            row[p["Slot"]] = f"{p['Name']} ({p['TEAM']} - {p['Salary']:.1f}M)"
        used = round(float(l["Salary"].sum()), 1)
        row.update(
            Salary_Used=used,
            Rem_Budget=round(budget - used, 1),
            Proj_Pts=round(float(l["Proj"].sum()), 1),
            Floor_Sum=round(float(l["Floor"].sum()), 1),
            Ceiling_Sum=round(float(l["Ceiling"].sum()), 1),
            Own_Sum_pct=round(float(l["Own%"].sum()), 1),
        )
        if idx < len(metrics):
            m = metrics.iloc[idx]
            for c in ["Sim_Mean", "Sim_P10", "Sim_P90", "Cash_Top45%", "Top10%", "Top1%"]:
                row[c] = m[c]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# PORTFOLIO RUNNER (one call per Cash / GPP)
# ---------------------------------------------------------
def run_portfolio(kind: str, full_raw: pd.DataFrame, keep_names: set, locked: list, cfg: dict) -> dict:
    lev = cfg.get("leverage", 0.0) if kind == "GPP" else 0.0
    max_own, chalk_thr, max_chalk = leverage_constraints(lev) if kind == "GPP" else (700.0, 30.0, ROSTER_SIZE)

    scored = compute_metrics_after_edit(
        full_raw, kind, cfg["blend"], cfg.get("env_boost", False),
        cfg["blowout_thr"], cfg["blowout_pct"], leverage_lambda(lev),
    )
    scored["Max_Exp%"] = scored["Max_Exp%"].fillna(float(cfg["max_exp"]))
    for c in ("Proj", "Floor", "Ceiling"):  # downstream tables show the adjusted numbers
        scored[c + "_Raw"] = scored[c]
        scored[c] = scored["Adj_" + c]

    pool = scored[
        scored["Name"].isin(keep_names)
        & ((scored["Min"] >= cfg["min_minutes"]) | scored["Name"].isin(locked))
    ].reset_index(drop=True)

    problems = diagnose_settings(
        pool, cfg["budget"], cfg["min_budget"], locked,
        cfg["stacks"] if kind == "GPP" else {}, cfg["max_team"],
        cfg["linked_pairs"], cfg["ban_pairs"],
    )
    if problems:
        return {"error": problems}

    n_final = cfg["n"]
    if kind == "GPP":
        n_gen = int(min(150, max(n_final, n_final * cfg["cand_mult"])))
        var = VARIETY[cfg["variety"]]
        randomness, team_share = var["randomness"], var["team_share"]
        min_unique_final = var["min_unique"]
        min_unique_gen = max(1, min_unique_final - 1) if n_gen > n_final else min_unique_final
    else:
        n_gen, randomness, team_share = n_final, 0.0, 0.0
        min_unique_final = min_unique_gen = 2

    lineups = generate_lineups(
        data=pool, num_lineups=n_gen, budget=cfg["budget"], min_budget=cfg["min_budget"],
        locked=locked, stacks=cfg["stacks"] if kind == "GPP" else {},
        bring_back_enabled=bool(cfg.get("bring_back", False)) and kind == "GPP",
        auto_bring_back_min=3 if (kind == "GPP" and cfg.get("bring_back", False)) else 0,
        max_team=cfg["max_team"], max_per_game=cfg["max_game"], min_unique=min_unique_gen,
        randomness=randomness, team_shock_share=team_share,
        max_lineup_own=max_own, max_chalk_count=max_chalk, chalk_threshold=chalk_thr,
        linked_pairs=cfg["linked_pairs"], ban_pairs=cfg["ban_pairs"],
        solver_time_limit=cfg["solver_time"], seed=cfg["seed"],
    )
    if not lineups:
        return {"error": ["No feasible lineups. Relax stacks, ownership limits, exposure caps or minimum salary."]}

    sims = simulate_player_scores(scored, cfg["num_sims"], cfg["rho_game"], cfg["rho_team"], cfg["seed"] + 1)
    field = simulate_field_matrix(scored, cfg["budget"], cfg["field_size"], cfg["seed"] + 2, keep_frac=cfg["field_keep"])
    thresholds = field_thresholds(field, sims)
    metrics = evaluate_lineups(lineups, scored, sims, thresholds)

    n_candidates = len(lineups)
    if kind == "GPP" and n_candidates > n_final:
        target = cfg["gpp_metric"] if thresholds else "Sim_Mean"
        order = list(np.lexsort((-metrics["Sim_Mean"].to_numpy(), -metrics[target].fillna(0).to_numpy())))
        caps = portfolio_max_counts(pool, locked, n_final)
        chosen = select_diverse(lineups, order, n_final, min_unique_final, caps)
        lineups = [lineups[i] for i in chosen]
        metrics = metrics.iloc[chosen].reset_index(drop=True)
        metrics["Lineup_#"] = np.arange(1, len(lineups) + 1)

    detailed = []
    for i, l in enumerate(lineups):
        t = l.copy()
        t.insert(0, "Lineup_#", i + 1)
        detailed.append(t)
    return {
        "kind": kind,
        "lineups": lineups,
        "metrics": metrics,
        "compact": build_compact_export(lineups, metrics, cfg["budget"]),
        "detailed": pd.concat(detailed, ignore_index=True),
        "budget": cfg["budget"],
        "n_candidates": n_candidates,
        "pool_size": len(pool),
        "gpp_metric": cfg.get("gpp_metric", "Cash_Top45%"),
        "field_ok": thresholds is not None,
        "preset": cfg["preset"],
    }


# ---------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------
def _safe_download_button(label: str, data: bytes, file_name: str, key: str):
    kwargs = dict(label=label, data=data, file_name=file_name, mime="text/csv", key=key)
    try:
        st.download_button(width="stretch", on_click="ignore", **kwargs)
    except TypeError:
        st.download_button(**kwargs)


CSS = """
<style>
.euro-hero { background: linear-gradient(135deg, #161B26 0%, #0B0E14 65%, #2A1205 100%);
  border: 1px solid #2A3241; border-left: 6px solid #FF5A00; border-radius: 12px;
  padding: 20px 26px; margin-bottom: 18px; }
.euro-title { font-size: 1.95rem; font-weight: 800; margin: 0 0 4px 0; letter-spacing: -0.5px; }
.euro-accent { color: #FF5A00; }
.euro-subtitle { color: #94A3B8; font-size: 0.92rem; margin-bottom: 12px; }
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; }
.euro-badge { background-color: rgba(255, 90, 0, 0.14); border: 1px solid rgba(255, 90, 0, 0.4);
  color: #FF8533; font-size: 0.75rem; font-weight: 600; padding: 3px 11px; border-radius: 999px;
  text-transform: uppercase; }
.lineup-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: 10px; margin: 12px 0 14px 0; }
.court-card { background: linear-gradient(180deg, #181E2C 0%, #10141D 100%);
  border: 1px solid #2A3241; border-radius: 10px; padding: 12px 10px; text-align: center; }
.slot-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.slot-pill { background-color: #FF5A00; color: #FFFFFF; font-weight: 800; font-size: 0.70rem;
  padding: 2px 8px; border-radius: 5px; }
.slot-pill-flex { background-color: #3B82F6; color: #FFFFFF; font-weight: 800; font-size: 0.70rem;
  padding: 2px 8px; border-radius: 5px; }
.team-chip { font-size: 0.68rem; font-weight: 800; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.5px; }
.p-name { font-weight: 700; font-size: 0.92rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; margin: 4px 0 2px 0; }
.p-matchup { font-size: 0.74rem; color: #94A3B8; margin-bottom: 8px; }
.p-stats { display: flex; justify-content: space-around; background-color: #0B0E14;
  border-radius: 6px; padding: 6px 4px; border: 1px solid #1E2636; }
.p-stat-label { font-size: 0.62rem; color: #64748B; text-transform: uppercase; }
.p-stat-val { font-size: 0.82rem; font-weight: 700; }
.p-stat-proj { font-size: 0.82rem; font-weight: 800; color: #FF5A00; }
.cap-bar-wrap { background-color: #0B0E14; border: 1px solid #262F40; border-radius: 8px;
  padding: 8px 12px; margin-bottom: 10px; }
.cap-bar-track { width: 100%; height: 8px; background-color: #1E2636; border-radius: 999px;
  overflow: hidden; margin-top: 5px; }
.cap-bar-fill { height: 100%; background: linear-gradient(90deg, #FF5A00 0%, #22C55E 100%);
  border-radius: 999px; }
.game-card { background: #161B26; border: 1px solid #2A3241; border-radius: 10px; padding: 16px; margin-bottom: 12px; }
.game-card-prime { background: linear-gradient(135deg, #161B26 0%, #2A1508 100%);
  border: 1px solid #FF5A00; border-radius: 10px; padding: 16px; margin-bottom: 12px; }
</style>
"""

HERO = """
<div class="euro-hero">
  <div class="euro-title">🏀 EUROLEAGUE <span class="euro-accent">PRO DFS SUITE</span></div>
  <div class="euro-subtitle">FanTeam 5+2 optimizer: cash and GPP portfolios, ownership leverage, simulated against the field</div>
  <div class="badge-row">
    <span class="euro-badge">PG • SG • SF • PF • C • 2 FLEX</span>
    <span class="euro-badge">No captains</span>
    <span class="euro-badge">Tie-break: remaining salary</span>
  </div>
</div>
"""


def player_card_html(p) -> str:
    slot = str(p["Slot"])
    pill = "slot-pill-flex" if slot.startswith("FLEX") else "slot-pill"
    bg, fg = team_colors(str(p["TEAM"]))
    name = _html.escape(str(p["Name"]))
    return _flat(
        f"""
        <div class="court-card">
          <div class="slot-header">
            <span class="{pill}">{_html.escape(slot)}</span>
            <span class="team-chip" style="background:{bg}; color:{fg};">{_html.escape(str(p["TEAM"]))}</span>
          </div>
          <div class="p-name" title="{name}">{name}</div>
          <div class="p-matchup">{p["Position"]} • vs {_html.escape(str(p["Opp"]))} • {p["Own%"]:.1f}% own</div>
          <div class="p-stats">
            <div><div class="p-stat-label">Sal</div><div class="p-stat-val">{p["Salary"]:.1f}M</div></div>
            <div><div class="p-stat-label">Proj</div><div class="p-stat-proj">{p["Proj"]:.1f}</div></div>
            <div><div class="p-stat-label">P10-P90</div><div class="p-stat-val">{p["Floor"]:.0f}-{p["Ceiling"]:.0f}</div></div>
          </div>
        </div>
        """
    )


def render_portfolio(kind: str, res: dict, key_prefix: str):
    lineups, metrics, budget = res["lineups"], res["metrics"], res["budget"]
    detailed = res["detailed"]

    top, d1, d2 = st.columns([2, 1, 1])
    with top:
        extra = ""
        if kind == "GPP" and res["n_candidates"] > len(lineups):
            target = [k for k, v in GPP_TARGETS.items() if v == res["gpp_metric"]]
            extra = f" — picked from {res['n_candidates']} candidates by simulated {target[0] if target else 'upside'}"
        st.success(f"{kind} · {res['preset']}: {len(lineups)} lineup(s){extra}")
        if not res["field_ok"]:
            st.warning("The simulated field could not be built, so field-based percentages are blank.")
    with d1:
        _safe_download_button("📥 Compact (1 row / lineup)", res["compact"].to_csv(index=False).encode("utf-8"),
                              f"euroleague_{kind.lower()}_compact.csv", f"{key_prefix}_dl_compact")
    with d2:
        _safe_download_button("📥 Full player rows", detailed.to_csv(index=False).encode("utf-8"),
                              f"euroleague_{kind.lower()}_detailed.csv", f"{key_prefix}_dl_detail")

    t_cards, t_table, t_expo = st.tabs(["🏀 Lineups", "⚡ Quick-entry table", "📊 Exposure & stacks"])

    with t_cards:
        for i, l in enumerate(lineups):
            used = float(l["Salary"].sum())
            rem = budget - used
            pct = min(100.0, used / budget * 100.0)
            m = metrics.iloc[i]
            stacks = ", ".join(f"{k} ({v})" for k, v in l["TEAM"].value_counts().items() if v >= 2)
            st.markdown(f"#### Lineup {i + 1} — stacks: `{stacks or 'balanced'}`")
            st.markdown(
                _flat(
                    f"""
                    <div class="cap-bar-wrap">
                      <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                        <span><b>Salary used:</b> {used:.1f}M / {budget:.1f}M</span>
                        <span style="color:#22C55E;"><b>Remaining (tie-break):</b> {rem:.1f}M</span>
                      </div>
                      <div class="cap-bar-track"><div class="cap-bar-fill" style="width:{pct:.1f}%;"></div></div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            c = st.columns(7)
            c[0].metric("Proj", f"{l['Proj'].sum():.1f}")
            c[1].metric("Sim mean", f"{m['Sim_Mean']:.1f}")
            c[2].metric("P10 / P90", f"{m['Sim_P10']:.0f} / {m['Sim_P90']:.0f}")
            c[3].metric("Total own", f"{l['Own%'].sum():.0f}%")
            c[4].metric("Beats 55% of field", "—" if pd.isna(m["Cash_Top45%"]) else f"{m['Cash_Top45%']:.0f}%")
            c[5].metric("Top 10%", "—" if pd.isna(m["Top10%"]) else f"{m['Top10%']:.1f}%")
            c[6].metric("Top 1%", "—" if pd.isna(m["Top1%"]) else f"{m['Top1%']:.1f}%")
            st.markdown(
                '<div class="lineup-grid">' + "".join(player_card_html(p) for _, p in l.iterrows()) + "</div>",
                unsafe_allow_html=True,
            )

    with t_table:
        st.dataframe(res["compact"], width="stretch", hide_index=True)

    with t_expo:
        e1, e2 = st.columns([2, 1])
        with e1:
            st.subheader("Player exposure vs. field ownership")
            exp = detailed.groupby(["Name", "Position", "TEAM", "Salary", "Proj", "Own%"]).size().reset_index(name="Lineups")
            exp["Portfolio_%"] = (exp["Lineups"] / len(lineups) * 100.0).round(1)
            exp["Leverage_%"] = (exp["Portfolio_%"] - exp["Own%"]).round(1)
            exp = exp.sort_values(["Portfolio_%", "Proj"], ascending=[False, False])
            st.dataframe(
                exp, width="stretch", hide_index=True,
                column_config={
                    "Portfolio_%": st.column_config.ProgressColumn("Portfolio %", format="%.1f%%", min_value=0.0, max_value=100.0),
                    "Own%": st.column_config.NumberColumn("Field own%", format="%.1f%%"),
                    "Leverage_%": st.column_config.NumberColumn("Leverage", format="%+.1f%%"),
                },
            )
        with e2:
            st.subheader("Team and game exposure")
            t = detailed.groupby("TEAM").size().reset_index(name="Spots")
            t["Avg_Per_Lineup"] = (t["Spots"] / len(lineups)).round(2)
            st.dataframe(t.sort_values("Spots", ascending=False), width="stretch", hide_index=True)
            g = detailed.groupby("Game_Key").size().reset_index(name="Spots")
            g["Avg_Per_Lineup"] = (g["Spots"] / len(lineups)).round(2)
            st.dataframe(g.sort_values("Spots", ascending=False), width="stretch", hide_index=True)


def advanced_settings(kind: str, preset: str, n: int) -> dict:
    """Advanced controls for one portfolio. Keys include the preset, so switching preset resets them."""
    p = PRESETS[kind][preset]
    k = f"{kind}_{preset}"
    label = "Floor weight (%)" if kind == "Cash" else "Ceiling weight (%)"
    c1, c2, c3 = st.columns(3)
    blend = c1.slider(label, 0, 100, int(p["blend"] * 100), key=f"{k}_blend") / 100.0
    min_minutes = c2.slider("Min projected minutes", 0.0, 35.0, float(p["min_minutes"]), 0.5, key=f"{k}_min")
    max_exp = c3.slider("Default max player exposure (%)", 10, 100, int(p["max_exp"]), 5, key=f"{k}_exp")
    c4, c5, c6 = st.columns(3)
    max_team = c4.slider("Max players per team", 1, 7, int(p["max_team"]), key=f"{k}_team")
    max_game = c5.slider("Max players per game", 2, 7, int(p["max_game"]), key=f"{k}_game")
    blowout_pct = c6.slider("Blowout haircut on big-favourite starters (%)", 0.0, 20.0, float(p["blowout_pct"]), 0.5, key=f"{k}_blow")
    out = dict(blend=blend, min_minutes=min_minutes, max_exp=max_exp, max_team=max_team,
               max_game=max_game, blowout_pct=blowout_pct, n=n, preset=preset)
    if kind == "GPP":
        c7, c8, c9 = st.columns(3)
        out["bring_back"] = c7.checkbox("Bring-back: stacked team needs ≥1 opponent", bool(p["bring_back"]), key=f"{k}_bb")
        out["env_boost"] = c8.checkbox("Boost high game totals (+3.5% max)", bool(p["env_boost"]), key=f"{k}_env")
        out["cand_mult"] = c9.slider("Candidate multiplier", 1, 8, 4, key=f"{k}_cand",
                                     help="Builds this many times more candidates, then keeps the best by simulation.")
    return out


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="EuroLeague Pro DFS Suite", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(_flat(HERO), unsafe_allow_html=True)

    # ---------------- SIDEBAR: the few controls that matter ----------------
    with st.sidebar:
        st.markdown("### 📂 Slate")
        uploaded = st.file_uploader("Projections (CSV / XLSX)", type=["csv", "xlsx"])
        is_sample = uploaded is None
        if is_sample:
            raw_df = get_sample_data()
        else:
            raw_df = pd.read_csv(uploaded, encoding="utf-8-sig") if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            st.success(f"Loaded {uploaded.name}")

        st.markdown("### 🎯 Build")
        build_type = st.radio("Portfolio", ["Cash", "GPP", "Both"], index=2, horizontal=True)
        want_cash, want_gpp = build_type in ("Cash", "Both"), build_type in ("GPP", "Both")

        n_cash = n_gpp = 0
        cash_preset, gpp_preset = "Balanced", "Balanced"
        leverage, variety, gpp_target = 0.0, "Med", "Large GPP (top 1%)"
        if want_cash:
            st.markdown("**Cash**")
            n_cash = st.slider("Cash lineups", 1, 10, 3)
            cash_preset = st.selectbox("Cash strategy", list(PRESETS["Cash"]), key="cash_preset")
            st.caption(PRESET_HELP["Cash"][cash_preset])
        if want_gpp:
            st.markdown("**GPP**")
            n_gpp = st.slider("GPP lineups", 1, 50, 15)
            gpp_preset = st.selectbox("GPP strategy", list(PRESETS["GPP"]), key="gpp_preset")
            st.caption(PRESET_HELP["GPP"][gpp_preset])
            gp = PRESETS["GPP"][gpp_preset]
            leverage = st.slider(
                "Ownership leverage", 0, 100, int(gp["leverage"] * 100), step=5, key=f"lev_{gpp_preset}",
                help="Higher = fades chalk harder: bigger ownership penalty, lower total-ownership cap, fewer chalk players per lineup.",
            ) / 100.0
            variety = st.select_slider("Lineup variety", ["Low", "Med", "High"], value=gp["variety"], key=f"var_{gpp_preset}")
            gpp_target = st.selectbox("Pick GPP lineups by", list(GPP_TARGETS), key="gpp_target")

    # ---------------- DATA PIPELINE ----------------
    try:
        df = normalize_dataframe(raw_df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    dropped_rows = df.attrs.get("dropped_rows", 0)

    if is_sample:
        st.warning("⚠️ **Demo slate active** — placeholder numbers. Upload today's projections in the sidebar before building real lineups.")
    if dropped_rows:
        st.caption(f"ℹ️ {dropped_rows} row(s) were ignored (salary of 0, unknown position or blank name).")

    report = validate_slate_data(df)
    auto_fix = True
    with st.expander(f"🔍 Data check — {report['total_issues']} anomaly(ies)", expanded=False):
        auto_fix = st.checkbox("Auto-repair Floor / Ceiling order and zero floors", value=True)
        a, b, c, d = st.columns(4)
        a.metric("Floor > Proj", len(report["floor_gt_proj"]))
        b.metric("Ceiling < Proj", len(report["ceil_lt_proj"]))
        c.metric("Floor ≤ 0", len(report["zero_floor"]))
        d.metric("Team minutes > 200", len(report["high_team_mins"]))
        for key, caption in [("floor_gt_proj", "Floor above Proj"), ("ceil_lt_proj", "Ceiling below Proj"),
                             ("zero_floor", "Floor at 0"), ("high_team_mins", "Teams with more than 200 listed minutes")]:
            if not report[key].empty:
                st.caption(caption)
                st.dataframe(report[key], width="stretch", hide_index=True)
    if auto_fix:
        df = repair_slate_anomalies(df)

    # ---------------- OWNERSHIP (estimated once, on the FULL slate) ----------------
    saved_params = load_ownership_params()
    with st.sidebar:
        with st.expander("⚙️ Ownership model"):
            sharp = st.slider("Chalk concentration", 0.5, 3.0, float(saved_params["sharpness"]), 0.1,
                              key=f"sharp_{saved_params['sharpness']}",
                              help="Fitted automatically from your calibration log.")
    df["Est_Own%"] = estimate_ownership(df, sharpness=sharp, w_val=saved_params["w_val"])
    csv_has_own = df["Own%"].sum() > 0
    own_source = "CSV column" if csv_has_own else f"Estimated (sharpness {sharp:.1f})"
    if not csv_has_own:
        df["Own%"] = df["Est_Own%"]

    # ---------------- TABS ----------------
    tab_opt, tab_slate, tab_own = st.tabs(["🛠️ Optimizer", "🏟️ Slate overview", "📈 Ownership & calibration"])

    # ---- Slate overview ----
    with tab_slate:
        st.subheader("Game environments")
        rows = []
        for g_key, g in df.groupby("Game_Key"):
            teams_in = sorted(g["TEAM"].unique())
            t1 = teams_in[0]
            t2 = teams_in[1] if len(teams_in) > 1 else g["Opp"].iloc[0]
            p1 = g.loc[g["TEAM"] == t1, "Team_Pts"].median()
            p2 = g.loc[g["TEAM"] == t2, "Team_Pts"].median() if t2 in g["TEAM"].values else g["Opp_Pts"].median()
            wp = g.loc[g["TEAM"] == t1, "Win_Prob"].median()
            rows.append(dict(A=t1, B=t2, PA=round(float(p1), 1), PB=round(float(p2), 1),
                             Total=round(float(p1 + p2), 1), Spread=round(abs(float(p1 - p2)), 1),
                             Fav=t1 if p1 >= p2 else t2, Pace=round(float(g["Pace"].median()), 1),
                             FavWin=round(float(max(wp, 1.0 - wp) * 100.0), 1)))
        games = pd.DataFrame(rows).sort_values("Total", ascending=False)
        cols = st.columns(min(3, max(1, len(games))))
        for i, (_, r) in enumerate(games.iterrows()):
            prime = i == 0
            badge = '<span class="euro-badge" style="float:right;">🔥 Best environment</span>' if prime else ""
            with cols[i % len(cols)]:
                st.markdown(
                    _flat(
                        f"""
                        <div class="{'game-card-prime' if prime else 'game-card'}">
                          {badge}
                          <div style="font-size:1.1rem; font-weight:800; margin-bottom:6px;">{r.A} ({r.PA}) vs {r.B} ({r.PB})</div>
                          <div style="color:#94A3B8; font-size:0.85rem; margin-bottom:10px;">
                            Favourite: <b>{r.Fav} (-{r.Spread})</b> • win prob <b>{r.FavWin}%</b></div>
                          <div class="p-stats">
                            <div><div class="p-stat-label">Total</div><div class="p-stat-proj">{r.Total}</div></div>
                            <div><div class="p-stat-label">Pace</div><div class="p-stat-val">{r.Pace}</div></div>
                            <div><div class="p-stat-label">Spread</div><div class="p-stat-val">{r.Spread}</div></div>
                          </div>
                        </div>
                        """
                    ),
                    unsafe_allow_html=True,
                )

    # ---- Ownership & calibration ----
    with tab_own:
        st.subheader("Ownership")
        st.caption(
            "FanTeam shows no ownership before lock, so lineups use the estimate above. "
            "After lock (or once some games have started), paste the real percentages here: "
            "they take over immediately and can be saved to improve the estimator for future slates."
        )
        o1, o2 = st.columns(2)
        with o1:
            pasted = st.text_area("Paste real ownership (Name, Own% — one per line)", height=180,
                                  placeholder="T.J. Shorts, 38.5%\nTremont Waters, 41%")
            pasted_file = st.file_uploader("…or upload a CSV with Name and Own%", type=["csv"], key="real_own_file")
            unlisted_low = st.checkbox("Treat players not listed as 0.5% owned", value=False)
        real_text = pasted or ""
        if pasted_file is not None:
            real_text += "\n" + pasted_file.getvalue().decode("utf-8-sig", errors="ignore")
        matched, unmatched = parse_real_ownership(real_text, df["Name"]) if real_text.strip() else ({}, [])
        with o2:
            if matched:
                st.success(f"Matched {len(matched)} player(s). Real ownership is now driving leverage and the simulated field.")
            if unmatched:
                st.warning("Could not match: " + ", ".join(unmatched[:15]) + (" …" if len(unmatched) > 15 else ""))
            slate_label = st.text_input("Slate label (for the calibration log)", value=f"EuroLeague {date.today().isoformat()}")
            if st.button("💾 Save to calibration log", disabled=not matched, width="stretch"):
                tmp = df[["Name", "TEAM", "Position", "Salary", "Proj", "Min", "Win_Prob", "Est_Own%"]].copy()
                tmp["Real_Own%"] = tmp["Name"].map(matched)
                if unlisted_low:
                    tmp["Real_Own%"] = tmp["Real_Own%"].fillna(0.5)
                log = append_calibration_log(tmp, slate_label)
                st.success(f"Saved. The log now holds {log['Slate'].nunique()} slate(s).")

        st.divider()
        st.subheader("Calibrate the estimator")
        log = read_calibration_log()
        hist = st.file_uploader("Add a past slate (CSV with projections plus Real_Own%)", type=["csv"], key="cal_upload")
        if hist is not None:
            try:
                h = normalize_dataframe(pd.read_csv(hist, encoding="utf-8-sig"))
                if "Real_Own%" in h.columns:
                    h["Est_Own%"] = estimate_ownership(h, sharpness=sharp, w_val=saved_params["w_val"])
                    log = append_calibration_log(h, Path(hist.name).stem)
                    st.success(f"Added {hist.name} to the log.")
                else:
                    st.info("Add a Real_Own% column to that file first.")
            except ValueError as exc:
                st.error(str(exc))
        if log.empty:
            st.info("No calibration data yet. Save a slate after lock to start improving the estimate.")
        else:
            st.caption(f"{log['Slate'].nunique()} slate(s), {int(log['Real_Own%'].notna().sum())} player rows with real ownership.")
            if st.button("🔧 Fit estimator to the log"):
                fit = fit_ownership_params(log)
                if fit:
                    save_ownership_params({"sharpness": fit["sharpness"], "w_val": fit["w_val"]})
                    st.success(f"Fitted: sharpness {fit['sharpness']}, value weight {fit['w_val']} (mean error {fit['mae']:.2f} points). Reload to apply.")
                else:
                    st.info("Need a slate with at least 5 real-ownership rows.")
            st.download_button("📥 Download calibration log", log.to_csv(index=False).encode("utf-8"),
                               "calibration_log.csv", "text/csv", key="dl_log")

    if matched:
        df["Own%"] = df["Name"].map(matched).fillna(0.5 if unlisted_low else df["Own%"])
        own_source = f"Real, pasted ({len(matched)} players)"

    # ---- Optimizer ----
    with tab_opt:
        st.caption(f"Ownership source: **{own_source}**")
        all_teams = sorted(df["TEAM"].unique().tolist())
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 🏢 Teams")
            included = st.multiselect("Include only (blank = all)", all_teams)
            excluded = st.multiselect("Exclude / tipped-off games", [t for t in all_teams if t not in included])
        names_all = sorted(df["Name"].tolist())
        with c2:
            st.markdown("##### 🔒 Locks and fades")
            locked = st.multiselect("Lock into every lineup (max 7)", names_all, max_selections=7,
                                    help="Late swap: lock your players from tipped-off games and exclude those teams here.")
            faded = st.multiselect("Fade (exclude)", [n for n in names_all if n not in locked])
        stack_rules, pair_a, pair_b = {}, [], []
        with c3:
            st.markdown("##### 🔥 GPP stacks")
            if want_gpp:
                stack_teams = st.multiselect("Force stack team(s)", all_teams)
                for t in stack_teams:
                    stack_rules[t] = st.number_input(f"Min players from {t}", 2, 7, 3, key=f"stack_{t}")
            else:
                st.caption("Stacks apply to GPP lineups.")

        with st.expander("🔗 Pair rules"):
            p1, p2 = st.columns(2)
            with p1:
                linked_sel = st.multiselect("Linked pair: both or neither", names_all, max_selections=2, key="linked_sel")
            with p2:
                ban_sel = st.multiselect("Never together", names_all, max_selections=2, key="ban_sel")
            linked_pairs = [tuple(linked_sel)] if len(linked_sel) == 2 else []
            ban_pairs = [tuple(ban_sel)] if len(ban_sel) == 2 else []

        # Settings that are shared by both portfolios + per-portfolio advanced
        with st.expander("⚙️ Advanced settings"):
            st.markdown("**Shared**")
            s1, s2, s3 = st.columns(3)
            budget = s1.number_input("Salary cap (M)", 40.0, 150.0, 75.0, 0.5)
            min_budget = s2.number_input("Minimum salary spent (M)", 0.0, float(budget), min(72.0, float(budget)), 0.5)
            blowout_thr = s3.slider("Big-favourite spread threshold (pts)", 4.0, 20.0, 10.0, 0.5)
            s4, s5, s6 = st.columns(3)
            num_sims = s4.select_slider("Simulation trials", [1000, 2500, 5000, 10000], value=2500)
            field_size = s5.select_slider("Simulated field lineups", [1000, 2000, 4000, 8000], value=4000)
            solver_time = s6.number_input("Solver time limit per lineup (s)", 1.0, 30.0, 5.0, 1.0)
            s7, s8, s9 = st.columns(3)
            rho_game = s7.slider("Same-game correlation", 0.0, 0.4, 0.10, 0.01)
            rho_team = s8.slider("Same-team correlation", 0.0, 0.4, 0.12, 0.01)
            field_keep = s9.slider("Field sharpness (keep top % by projection)", 5, 100, 15, 5,
                                   help="Real entrants are sharper than random picks. Lower = tougher simulated field.") / 100.0
            seed = int(st.number_input("Random seed", 0, 9999, 42, 1))
            cash_adv = gpp_adv = None
            if want_cash:
                st.markdown(f"**Cash — {cash_preset}**")
                cash_adv = advanced_settings("Cash", cash_preset, n_cash)
            if want_gpp:
                st.markdown(f"**GPP — {gpp_preset}**")
                gpp_adv = advanced_settings("GPP", gpp_preset, n_gpp)

        # Pool = everything not excluded/faded (locked players always stay)
        keep = ~df["TEAM"].isin(excluded) & ~df["Name"].isin(faded)
        if included:
            keep &= df["TEAM"].isin(included)
        keep |= df["Name"].isin(locked)
        keep_names = set(df.loc[keep, "Name"])
        master = df[df["Name"].isin(keep_names)].reset_index(drop=True)

        with st.expander(f"📋 Edit player pool ({len(master)} players)"):
            st.caption("Edit Salary, Proj, Floor, Ceiling, Own% or Max_Exp%. Everything derived from them is recalculated when you build. Leave Max_Exp% blank to use the strategy default.")
            edit_cols = ["Name", "Position", "TEAM", "Opp", "Salary", "Proj", "Floor", "Ceiling", "Own%", "Max_Exp%", "Min", "Value", "Favored_By"]
            editor_key = "pool_editor_" + hashlib.md5(",".join(master["Name"]).encode()).hexdigest()[:8]
            edited = st.data_editor(
                master[edit_cols], width="stretch", hide_index=True, num_rows="fixed", key=editor_key,
                disabled=["Name", "Position", "TEAM", "Opp", "Min", "Value", "Favored_By"],
                column_config={
                    "Salary": st.column_config.NumberColumn("Salary", format="%.1f"),
                    "Proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                    "Floor": st.column_config.NumberColumn("Floor (P10)", format="%.1f"),
                    "Ceiling": st.column_config.NumberColumn("Ceiling (P90)", format="%.1f"),
                    "Own%": st.column_config.NumberColumn("Own%", format="%.1f", min_value=0.0, max_value=100.0),
                    "Max_Exp%": st.column_config.NumberColumn("Max exp%", format="%.0f", min_value=0.0, max_value=100.0),
                },
            )

        # Apply edits back onto the full slate (the simulated field always uses everyone)
        full_raw = df.copy().set_index("Name")
        edited_idx = edited.set_index("Name")
        for col in ["Salary", "Proj", "Floor", "Ceiling", "Own%", "Max_Exp%"]:
            full_raw.loc[edited_idx.index, col] = edited_idx[col].to_numpy()
        full_raw = full_raw.reset_index()

        shared = dict(budget=float(budget), min_budget=float(min_budget), blowout_thr=float(blowout_thr),
                      num_sims=int(num_sims), field_size=int(field_size), solver_time=float(solver_time),
                      rho_game=float(rho_game), rho_team=float(rho_team), field_keep=float(field_keep), seed=seed,
                      linked_pairs=linked_pairs, ban_pairs=ban_pairs, stacks=stack_rules)

        if st.button("🚀 BUILD PORTFOLIO", type="primary", width="stretch"):
            results = {}
            with st.spinner("Solving lineups, simulating outcomes and the field…"):
                if want_cash:
                    results["Cash"] = run_portfolio("Cash", full_raw, keep_names, locked, {**shared, **cash_adv})
                if want_gpp:
                    cfg = {**shared, **gpp_adv, "leverage": leverage, "variety": variety, "gpp_metric": GPP_TARGETS[gpp_target]}
                    results["GPP"] = run_portfolio("GPP", full_raw, keep_names, locked, cfg)
            st.session_state["opt_results"] = results

        saved = st.session_state.get("opt_results")
        if saved:
            kinds = list(saved.keys())
            tabs = st.tabs([f"{'🛡️' if k == 'Cash' else '🎯'} {k}" for k in kinds]) if len(kinds) > 1 else [st.container()]
            for tab, kind in zip(tabs, kinds):
                with tab:
                    res = saved[kind]
                    if "error" in res:
                        for msg in res["error"]:
                            st.error(f"{kind}: {msg}")
                    else:
                        render_portfolio(kind, res, kind.lower())


if __name__ == "__main__":
    main()
