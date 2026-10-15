# cspell:disable
import io
import math

import numpy as np
import pandas as pd
import pulp
import streamlit as st

# ---------------------------------------------------------
# EUROLEAGUE TEAM COLOUR PALETTE
# ---------------------------------------------------------
TEAM_COLORS = {
    "OLY": ("#C8102E", "#FFFFFF"),
    "PAO": ("#007A33", "#FFFFFF"),
    "RMD": ("#FEBE10", "#0B0E14"),
    "FBB": ("#002B5C", "#F6EB61"),
    "BAR": ("#A50044", "#004D98"),
    "MON": ("#E30613", "#FFFFFF"),
    "PAR": ("#222222", "#FFFFFF"),
    "VIR": ("#111111", "#FFFFFF"),
    "PBB": ("#1D252D", "#00A3E0"),
    "HTA": ("#D00027", "#FFFFFF"),
    "BJK": ("#000000", "#FFFFFF"),
    "BAY": ("#DC052D", "#FFFFFF"),
    "ZAL": ("#006233", "#FFFFFF"),
    "EA7": ("#E30613", "#FFFFFF"),
    "ASV": ("#111111", "#FFFFFF"),
    "BKN": ("#002F6C", "#C8102E"),
    "MTA": ("#FCE300", "#003893"),
    "RED": ("#E30613", "#FFFFFF"),
    "ALB": ("#FFCD00", "#002B5C"),
}

REQUIRED_POSITIONS = ["PG", "SG", "SF", "PF", "C"]
SLOT_LABELS = ["PG", "SG", "SF", "PF", "C", "FLEX 1", "FLEX 2"]


# ---------------------------------------------------------
# PURE HELPER & MODEL FUNCTIONS (Importable by pytest)
# ---------------------------------------------------------
def get_sample_data() -> pd.DataFrame:
    """Returns clearly labeled demo data with 6 EuroLeague teams (3 games)."""
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


def normalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names, types, matchup keys, and spreads."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {
        "name": "Name",
        "player": "Name",
        "position": "Position",
        "pos": "Position",
        "team": "TEAM",
        "opp": "Opp",
        "salary": "Salary",
        "proj": "Proj",
        "floor": "Floor",
        "ceiling": "Ceiling",
        "value": "Value",
        "min": "Min",
        "team_pts": "Team_Pts",
        "pace": "Pace",
        "win_prob": "Win_Prob",
        "own%": "Own%",
        "own": "Own%",
        "ownership": "Own%",
        "max_exp%": "Max_Exp%",
    }
    df.rename(columns=lambda x: col_map.get(x.lower(), x), inplace=True)

    numeric_cols = [
        "Salary",
        "Proj",
        "Floor",
        "Ceiling",
        "Value",
        "Min",
        "Team_Pts",
        "Pace",
        "Win_Prob",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "Own%" not in df.columns:
        df["Own%"] = 0.0
    else:
        df["Own%"] = (
            df["Own%"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

    if "Max_Exp%" not in df.columns:
        df["Max_Exp%"] = 100.0
    else:
        df["Max_Exp%"] = (
            pd.to_numeric(df["Max_Exp%"], errors="coerce")
            .fillna(100.0)
            .clip(0.0, 100.0)
        )

    df["Name"] = df["Name"].astype(str).str.strip()
    df["Position"] = df["Position"].astype(str).str.upper().str.strip()
    df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip()
    df["Opp"] = df["Opp"].astype(str).str.upper().str.strip()

    df["Game_Key"] = df.apply(
        lambda r: " vs ".join(sorted([str(r["TEAM"]), str(r["Opp"])])), axis=1
    )

    team_pts_map = df.groupby("TEAM")["Team_Pts"].median().to_dict()
    df["Opp_Pts"] = df["Opp"].map(team_pts_map).fillna(df["Team_Pts"])
    df["Game_Total"] = (df["Team_Pts"] + df["Opp_Pts"]).round(1)
    df["Favored_By"] = (df["Team_Pts"] - df["Opp_Pts"]).round(1)
    df["Spread"] = (-df["Favored_By"]).round(1)

    return df


def validate_slate_data(df: pd.DataFrame) -> dict:
    """Checks for data anomalies: Floor > Proj, Ceiling < Proj, Floor == 0, Team Min > 200."""
    floor_gt_proj = df[df["Floor"] > df["Proj"]][
        ["Name", "TEAM", "Position", "Floor", "Proj", "Ceiling"]
    ]
    ceil_lt_proj = df[df["Ceiling"] < df["Proj"]][
        ["Name", "TEAM", "Position", "Floor", "Proj", "Ceiling"]
    ]
    zero_floor = df[(df["Proj"] > 0) & (df["Floor"] <= 0)][
        ["Name", "TEAM", "Position", "Floor", "Proj", "Ceiling"]
    ]
    team_mins = (
        df.groupby("TEAM")["Min"].sum().reset_index(name="Total_Team_Min").round(1)
    )
    high_team_mins = team_mins[team_mins["Total_Team_Min"] > 200.0]

    return {
        "floor_gt_proj": floor_gt_proj,
        "ceil_lt_proj": ceil_lt_proj,
        "zero_floor": zero_floor,
        "high_team_mins": high_team_mins,
        "total_issues": (
            len(floor_gt_proj)
            + len(ceil_lt_proj)
            + len(zero_floor)
            + len(high_team_mins)
        ),
    }


def repair_slate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-repairs Floor > Proj, Ceiling < Proj, and Floor <= 0 anomalies."""
    fixed = df.copy()
    zero_mask = (fixed["Proj"] > 0) & (fixed["Floor"] <= 0)
    fixed.loc[zero_mask, "Floor"] = (fixed.loc[zero_mask, "Proj"] * 0.65).round(1)
    fixed["Floor"] = np.minimum(fixed["Floor"], fixed["Proj"])
    fixed["Ceiling"] = np.maximum(fixed["Ceiling"], fixed["Proj"])
    return fixed


def estimate_ownership(
    df: pd.DataFrame,
    sharpness: float = 1.6,
    w_val: float = 0.40,
    w_proj: float = 0.35,
    w_min: float = 0.15,
    w_win: float = 0.10,
) -> pd.Series:
    """
    Stage 2 Ownership Model: Softmax of z-scored Value, Proj, Min, Win_Prob
    with iterative cap redistribution so the slate sums to 700% (7 roster spots).
    """
    if len(df) == 0:
        return pd.Series(dtype=float)

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    val_s = df["Proj"] / df["Salary"].replace(0, np.nan).fillna(10.0)
    z_total = (
        w_val * _zscore(val_s)
        + w_proj * _zscore(df["Proj"])
        + w_min * _zscore(df["Min"])
        + w_win * _zscore(df["Win_Prob"])
    )

    logits = (z_total * sharpness).clip(-15.0, 15.0)
    weights = np.exp(logits - logits.max())
    weights = np.where(df["Min"] > 0, weights, weights * 0.05)

    raw_own = pd.Series(700.0 * (weights / (weights.sum() + 1e-9)), index=df.index)

    max_cap = 75.0
    for _ in range(10):
        over = raw_own > max_cap
        if not over.any():
            break
        raw_own[over] = max_cap
        rem_target = 700.0 - float(raw_own[over].sum())
        under = ~over
        under_sum = float(raw_own[under].sum())
        if under.any() and under_sum > 0:
            raw_own[under] = raw_own[under] * (rem_target / under_sum)

    return raw_own.clip(0.1, max_cap).round(1)


def compute_metrics_after_edit(
    df: pd.DataFrame,
    contest_mode: str,
    blend_weight: float,
    env_boost: bool,
    blowout_threshold: float,
    blowout_penalty_pct: float,
    own_penalty_lambda: float,
    use_estimated_own: bool,
    own_sharpness: float,
) -> pd.DataFrame:
    """
    Recomputes Value, Blowout adjustments, Ownership, and Opt_Score
    AFTER st.data_editor so user edits to Proj/Floor/Ceiling/Salary take effect.
    """
    out = df.copy()
    out["Value"] = (out["Proj"] / out["Salary"].replace(0, np.nan).fillna(1.0)).round(2)

    blowout_mult = np.where(
        (out["Favored_By"] > blowout_threshold) & (out["Min"] >= 24.0),
        1.0 - (blowout_penalty_pct / 100.0),
        1.0,
    )
    out["Adj_Proj"] = (out["Proj"] * blowout_mult).round(2)
    out["Adj_Ceiling"] = (out["Ceiling"] * blowout_mult).round(2)

    if use_estimated_own or (out["Own%"].sum() == 0):
        out["Own%"] = estimate_ownership(out, sharpness=own_sharpness)

    if contest_mode == "Cash Game (Floor / Safety)":
        base_score = ((1.0 - blend_weight) * out["Adj_Proj"]) + (
            blend_weight * out["Floor"]
        )
    elif contest_mode == "GPP Tournament (Ceiling / Upside)":
        base_score = ((1.0 - blend_weight) * out["Adj_Proj"]) + (
            blend_weight * out["Adj_Ceiling"]
        )
        if env_boost and out["Game_Total"].max() > 0:
            tot_min = out["Game_Total"].min()
            tot_max = out["Game_Total"].max()
            env_norm = (out["Game_Total"] - tot_min) / (tot_max - tot_min + 1e-6)
            base_score = base_score * (1.0 + 0.035 * env_norm)
        base_score = base_score - (own_penalty_lambda * out["Own%"])
    else:
        base_score = out["Adj_Proj"]

    out["Opt_Score"] = base_score.round(2)
    return out


def assign_roster_slots(lineup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns the 7 players into exact FanTeam slots:
    PG, SG, SF, PF, C, FLEX 1, FLEX 2.
    """
    df = lineup_df.sort_values(by=["Salary", "Proj"], ascending=[False, False]).copy()
    assigned_indices = {}
    used = set()

    for pos in REQUIRED_POSITIONS:
        candidates = df[(df["Position"] == pos) & (~df.index.isin(used))]
        if not candidates.empty:
            chosen_idx = candidates.index[0]
            assigned_indices[pos] = chosen_idx
            used.add(chosen_idx)

    remaining = [idx for idx in df.index if idx not in used]
    ordered_rows = []
    for pos in REQUIRED_POSITIONS:
        if pos in assigned_indices:
            row = df.loc[assigned_indices[pos]].copy()
            row["Slot"] = pos
            ordered_rows.append(row)

    for flex_num, idx in enumerate(remaining, start=1):
        row = df.loc[idx].copy()
        row["Slot"] = f"FLEX {flex_num}"
        ordered_rows.append(row)

    result = pd.DataFrame(ordered_rows).reset_index(drop=True)
    cols = ["Slot"] + [c for c in result.columns if c != "Slot"]
    return result[cols]


def _make_binary_var(name: str):
    """Compatible binary variable constructor across PuLP versions."""
    try:
        return pulp.LpVariable(name, lowBound=0, upBound=1, cat="Binary")
    except TypeError:
        return pulp.LpVariable(name, 0, 1, pulp.LpBinary)


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
    force_pairs: list,
    ban_pairs: list,
    solver_time_limit: float = 5.0,
    seed: int = 42,
) -> list:
    """
    Builds the MILP model ONCE and adds uniqueness cuts & dynamic exposure
    bans incrementally across lineups 1..N.
    """
    if len(data) < 7:
        return []

    for pos in REQUIRED_POSITIONS:
        if pos not in data["Position"].unique():
            return []

    rng = np.random.default_rng(seed)

    prob = pulp.LpProblem("FanTeam_EuroLeague_Optimizer", pulp.LpMaximize)
    player_vars = {i: _make_binary_var(f"Player_{i}") for i in data.index}

    # 1. Roster Size == 7
    prob += (
        pulp.lpSum([player_vars[i] for i in data.index]) == 7,
        "Roster_Size_7",
    )

    # 2. Salary Cap & Floor
    prob += (
        pulp.lpSum([data.loc[i, "Salary"] * player_vars[i] for i in data.index])
        <= budget,
        "Max_Salary_Cap",
    )
    prob += (
        pulp.lpSum([data.loc[i, "Salary"] * player_vars[i] for i in data.index])
        >= min_budget,
        "Min_Salary_Floor",
    )

    # 3. Positional Minimums (PG, SG, SF, PF, C >= 1)
    for pos in REQUIRED_POSITIONS:
        prob += (
            pulp.lpSum(
                [player_vars[i] for i in data.index if data.loc[i, "Position"] == pos]
            )
            >= 1,
            f"Min_Pos_{pos}",
        )

    # 4. Max Players per Team
    for team in data["TEAM"].unique():
        prob += (
            pulp.lpSum(
                [player_vars[i] for i in data.index if data.loc[i, "TEAM"] == team]
            )
            <= max_team,
            f"Max_Team_{team}",
        )

    # 5. Max Players per Game (Matchup)
    if "Game_Key" in data.columns and max_per_game < 7:
        for g_key in data["Game_Key"].unique():
            prob += (
                pulp.lpSum(
                    [
                        player_vars[i]
                        for i in data.index
                        if data.loc[i, "Game_Key"] == g_key
                    ]
                )
                <= max_per_game,
                f"Max_Game_{g_key.replace(' ', '_')}",
            )

    # 6. Locked / Late-Swap Frozen Players
    for locked_name in locked:
        locked_indices = data[data["Name"] == locked_name].index.tolist()
        for idx in locked_indices:
            prob += player_vars[idx] == 1, f"Lock_{idx}"

    # 7. Explicit Team Stacking Rules + Bring-Back Rule
    for team, min_count in stacks.items():
        prob += (
            pulp.lpSum(
                [player_vars[i] for i in data.index if data.loc[i, "TEAM"] == team]
            )
            >= min_count,
            f"Stack_{team}",
        )
        if bring_back_enabled:
            opp_teams = data.loc[data["TEAM"] == team, "Opp"].unique().tolist()
            opp_indices = data[data["TEAM"].isin(opp_teams)].index.tolist()
            if opp_indices:
                prob += (
                    pulp.lpSum([player_vars[i] for i in opp_indices]) >= 1,
                    f"BringBack_Explicit_{team}",
                )

    # 8. Automatic Bring-Back on Any Solver-Created Stack >= auto_bring_back_min
    if bring_back_enabled and auto_bring_back_min >= 2:
        for team in data["TEAM"].unique():
            team_indices = data[data["TEAM"] == team].index.tolist()
            opp_teams = data.loc[data["TEAM"] == team, "Opp"].unique().tolist()
            opp_indices = data[data["TEAM"].isin(opp_teams)].index.tolist()
            if len(team_indices) >= auto_bring_back_min and opp_indices:
                stack_ind = _make_binary_var(f"StackInd_{team}")
                prob += (
                    pulp.lpSum([player_vars[i] for i in team_indices])
                    - (auto_bring_back_min - 1)
                    <= 7 * stack_ind,
                    f"AutoStackDetect_{team}",
                )
                prob += (
                    pulp.lpSum([player_vars[i] for i in opp_indices]) >= stack_ind,
                    f"AutoBringBack_{team}",
                )

    # 9. Ownership Constraints (Max Lineup Own% and Max Chalk Players)
    if "Own%" in data.columns and max_lineup_own < 700.0:
        prob += (
            pulp.lpSum([data.loc[i, "Own%"] * player_vars[i] for i in data.index])
            <= max_lineup_own,
            "Max_Lineup_Ownership",
        )

    if "Own%" in data.columns and max_chalk_count < 7:
        chalk_indices = data[data["Own%"] >= chalk_threshold].index.tolist()
        if chalk_indices:
            prob += (
                pulp.lpSum([player_vars[i] for i in chalk_indices]) <= max_chalk_count,
                "Max_Chalk_Players",
            )

    # 10. Force-Pair Rules (Roster both players together as an active unit)
    for pair_idx, (p_a, p_b) in enumerate(force_pairs):
        idx_a = data[data["Name"] == p_a].index.tolist()
        idx_b = data[data["Name"] == p_b].index.tolist()
        if idx_a and idx_b:
            prob += (
                player_vars[idx_a[0]] - player_vars[idx_b[0]] == 0,
                f"ForcePair_Link_{pair_idx}",
            )
            prob += (
                player_vars[idx_a[0]] == 1,
                f"ForcePair_Active_{pair_idx}",
            )

    # 11. Ban-Pair Rules
    for pair_idx, (p_a, p_b) in enumerate(ban_pairs):
        idx_a = data[data["Name"] == p_a].index.tolist()
        idx_b = data[data["Name"] == p_b].index.tolist()
        if idx_a and idx_b:
            prob += (
                player_vars[idx_a[0]] + player_vars[idx_b[0]] <= 1,
                f"BanPair_{pair_idx}",
            )

    max_exp_series = (
        data["Max_Exp%"]
        if "Max_Exp%" in data.columns
        else pd.Series(100.0, index=data.index)
    )
    forced_names = set(locked)
    for p_a, p_b in force_pairs:
        forced_names.add(p_a)
        forced_names.add(p_b)

    max_allowed_counts = {}
    for i in data.index:
        if data.loc[i, "Name"] in forced_names:
            max_allowed_counts[i] = num_lineups
        else:
            pct = float(max_exp_series.loc[i])
            if pct <= 0.0:
                max_allowed_counts[i] = 0
            elif pct >= 100.0:
                max_allowed_counts[i] = num_lineups
            else:
                max_allowed_counts[i] = max(
                    1, math.floor((pct / 100.0) * num_lineups + 1e-9)
                )

    exposure_counts = {i: 0 for i in data.index}
    banned_by_exposure = set()
    lineups = []

    try:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time_limit)
    except TypeError:
        solver = pulp.PULP_CBC_CMD(msg=False)

    for n in range(num_lineups):
        for i in data.index:
            if (
                i not in banned_by_exposure
                and exposure_counts[i] >= max_allowed_counts[i]
            ):
                prob += player_vars[i] == 0, f"ExpBan_P{i}_L{n}"
                banned_by_exposure.add(i)

        if randomness > 0 and n > 0:
            teams = data["TEAM"].unique()
            games = data["Game_Key"].unique()
            t_shocks = {
                t: rng.normal(0.0, randomness * team_shock_share) for t in teams
            }
            g_shocks = {g: rng.normal(0.0, randomness * 0.25) for g in games}
            ind_std = randomness * max(0.1, 1.0 - team_shock_share * 0.5)
            p_shocks = rng.normal(0.0, ind_std, size=len(data))

            total_mult = (
                1.0
                + data["TEAM"].map(t_shocks).to_numpy()
                + data["Game_Key"].map(g_shocks).to_numpy()
                + p_shocks
            )
            current_scores = data["Opt_Score"].to_numpy() * np.clip(
                total_mult, 0.5, 1.6
            )
        else:
            current_scores = data["Opt_Score"].to_numpy()

        prob.setObjective(
            pulp.lpSum(
                [
                    float(current_scores[pos_idx]) * player_vars[df_idx]
                    for pos_idx, df_idx in enumerate(data.index)
                ]
            )
        )

        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            break

        selected_indices = [
            i
            for i in data.index
            if player_vars[i].varValue is not None and player_vars[i].varValue > 0.5
        ]
        if len(selected_indices) != 7:
            break

        for i in selected_indices:
            exposure_counts[i] += 1

        prob += (
            pulp.lpSum([player_vars[i] for i in selected_indices]) <= (7 - min_unique),
            f"Uniqueness_Cut_{n}",
        )

        raw_lineup = data.loc[selected_indices].copy()
        slotted_lineup = assign_roster_slots(raw_lineup)
        lineups.append(slotted_lineup)

    return lineups


def run_monte_carlo_and_field_sim(
    pool_df: pd.DataFrame,
    lineups: list,
    budget: float,
    target_score: float,
    num_sims: int = 2500,
    num_field_lineups: int = 400,
    seed: int = 101,
) -> pd.DataFrame:
    """
    Vectorized Correlated Monte Carlo & Field Simulator.
    - Treats Floor as P10 (z = -1.2816) and Ceiling as P90 (z = +1.2816).
    - Correlates players via shared Game factor (pace/OT) and Team factor.
    - Samples realistic ownership-weighted opponent field lineups and computes
      Simulated Mean, P10, P90, P(Score >= Target), Cash Rate (Top 17%), and Top-1% Rate.
    """
    if not lineups or len(pool_df) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    n_players = len(pool_df)
    name_to_idx = {name: i for i, name in enumerate(pool_df["Name"])}

    mu = pool_df["Adj_Proj" if "Adj_Proj" in pool_df.columns else "Proj"].to_numpy()
    sigma_down = np.maximum(0.5, (mu - pool_df["Floor"].to_numpy()) / 1.28155)
    sigma_up = np.maximum(0.5, (pool_df["Ceiling"].to_numpy() - mu) / 1.28155)

    game_codes, _ = pd.factorize(pool_df["Game_Key"])
    team_codes, _ = pd.factorize(pool_df["TEAM"])
    n_games = max(1, game_codes.max() + 1)
    n_teams = max(1, team_codes.max() + 1)

    z_game = rng.standard_normal((n_games, num_sims))
    z_team = rng.standard_normal((n_teams, num_sims))
    z_ind = rng.standard_normal((n_players, num_sims))

    z_total = (
        math.sqrt(0.20) * z_game[game_codes, :]
        + math.sqrt(0.15) * z_team[team_codes, :]
        + math.sqrt(0.65) * z_ind
    )

    player_sim_scores = np.where(
        z_total < 0,
        np.maximum(0.0, mu[:, None] + z_total * sigma_down[:, None]),
        mu[:, None] + z_total * sigma_up[:, None],
    )

    n_user = len(lineups)
    user_matrix = np.zeros((n_user, n_players), dtype=np.float32)
    for l_idx, l_df in enumerate(lineups):
        for p_name in l_df["Name"]:
            if p_name in name_to_idx:
                user_matrix[l_idx, name_to_idx[p_name]] = 1.0

    user_sim_scores = user_matrix @ player_sim_scores

    own_weights = np.maximum(pool_df["Own%"].to_numpy(), 0.2)
    salaries = pool_df["Salary"].to_numpy()
    pos_indices = {
        pos: np.where(pool_df["Position"].to_numpy() == pos)[0]
        for pos in REQUIRED_POSITIONS
    }

    field_matrix = np.zeros((num_field_lineups, n_players), dtype=np.float32)
    valid_field = 0
    attempts = 0
    max_attempts = num_field_lineups * 6

    while valid_field < num_field_lineups and attempts < max_attempts:
        attempts += 1
        chosen = []
        for pos in REQUIRED_POSITIONS:
            cands = pos_indices[pos]
            w = own_weights[cands]
            w = w / w.sum()
            chosen.append(int(rng.choice(cands, p=w)))

        rem_cands = np.setdiff1d(np.arange(n_players), chosen)
        if len(rem_cands) < 2:
            break
        w_rem = own_weights[rem_cands]
        w_rem = w_rem / w_rem.sum()
        flex_picks = rng.choice(rem_cands, size=2, replace=False, p=w_rem)
        chosen.extend([int(x) for x in flex_picks])

        if salaries[chosen].sum() <= budget:
            field_matrix[valid_field, chosen] = 1.0
            valid_field += 1

    if valid_field > 0:
        field_sim_scores = field_matrix[:valid_field] @ player_sim_scores
    else:
        field_sim_scores = user_sim_scores

    cash_cutoff = np.percentile(field_sim_scores, 83.0, axis=0)
    top1_cutoff = np.percentile(field_sim_scores, 99.0, axis=0)

    records = []
    for l_idx in range(n_user):
        scores = user_sim_scores[l_idx, :]
        records.append(
            {
                "Lineup_#": l_idx + 1,
                "Sim_Mean": round(float(scores.mean()), 1),
                "Sim_P10": round(float(np.percentile(scores, 10.0)), 1),
                "Sim_P90": round(float(np.percentile(scores, 90.0)), 1),
                "Hit_Target_%": round(
                    float((scores >= target_score).mean() * 100.0), 1
                ),
                "Cash_Top17%": round(float((scores >= cash_cutoff).mean() * 100.0), 1),
                "GPP_Top1%": round(float((scores >= top1_cutoff).mean() * 100.0), 1),
            }
        )

    return pd.DataFrame(records)


def build_compact_export(
    lineups: list, sim_df: pd.DataFrame, budget: float
) -> pd.DataFrame:
    """
    Builds a 1-row-per-lineup CSV in exact PG, SG, SF, PF, C, FLEX 1, FLEX 2 order
    for rapid entry on FanTeam.
    """
    rows = []
    for idx, l_df in enumerate(lineups):
        row = {"Lineup_#": idx + 1}
        for _, p_row in l_df.iterrows():
            slot = p_row["Slot"]
            row[slot] = f"{p_row['Name']} ({p_row['TEAM']} - {p_row['Salary']:.1f}M)"

        tot_sal = round(float(l_df["Salary"].sum()), 1)
        row["Salary_Used"] = tot_sal
        row["Rem_Budget"] = round(budget - tot_sal, 1)
        row["Proj_Pts"] = round(float(l_df["Proj"].sum()), 1)
        row["Floor_Sum"] = round(float(l_df["Floor"].sum()), 1)
        row["Ceiling_Sum"] = round(float(l_df["Ceiling"].sum()), 1)
        row["Own_Sum_%"] = round(float(l_df["Own%"].sum()), 1)

        if not sim_df.empty and idx < len(sim_df):
            s_row = sim_df.iloc[idx]
            row["Sim_Mean"] = s_row["Sim_Mean"]
            row["Sim_P10"] = s_row["Sim_P10"]
            row["Sim_P90"] = s_row["Sim_P90"]
            row["Cash_Top17%"] = s_row["Cash_Top17%"]
            row["GPP_Top1%"] = s_row["GPP_Top1%"]

        rows.append(row)
    return pd.DataFrame(rows)


def _safe_download_button(label: str, data: bytes, file_name: str, key: str):
    """Uses on_click='ignore' on modern Streamlit with fallback for older versions."""
    try:
        st.download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime="text/csv",
            width="stretch",
            on_click="ignore",
            key=key,
        )
    except TypeError:
        st.download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime="text/csv",
            width="stretch",
            key=key,
        )


# ---------------------------------------------------------
# STREAMLIT APPLICATION UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="EuroLeague Pro DFS Optimizer | FanTeam",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .euro-hero {
            background: linear-gradient(135deg, #161B26 0%, #0B0E14 65%, #2A1205 100%);
            border: 1px solid #2A3241;
            border-left: 6px solid #FF5A00;
            border-radius: 12px;
            padding: 20px 26px;
            margin-bottom: 18px;
        }
        .euro-title {
            font-size: 1.95rem;
            font-weight: 800;
            margin: 0 0 4px 0;
            letter-spacing: -0.5px;
        }
        .euro-accent { color: #FF5A00; }
        .euro-subtitle { color: #94A3B8; font-size: 0.92rem; margin-bottom: 12px; }
        .badge-row { display: flex; flex-wrap: wrap; gap: 8px; }
        .euro-badge {
            background-color: rgba(255, 90, 0, 0.14);
            border: 1px solid rgba(255, 90, 0, 0.4);
            color: #FF8533;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 11px;
            border-radius: 999px;
            text-transform: uppercase;
        }

        /* Responsive 7-Player Court Grid */
        .lineup-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            gap: 10px;
            margin: 12px 0 14px 0;
        }
        .court-card {
            background: linear-gradient(180deg, #181E2C 0%, #10141D 100%);
            border: 1px solid #2A3241;
            border-radius: 10px;
            padding: 12px 10px;
            text-align: center;
        }
        .slot-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .slot-pill {
            background-color: #FF5A00;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 0.70rem;
            padding: 2px 8px;
            border-radius: 5px;
        }
        .slot-pill-flex {
            background-color: #3B82F6;
            color: #FFFFFF;
            font-weight: 800;
            font-size: 0.70rem;
            padding: 2px 8px;
            border-radius: 5px;
        }
        .team-chip {
            font-size: 0.68rem;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }
        .p-name {
            font-weight: 700;
            font-size: 0.92rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin: 4px 0 2px 0;
        }
        .p-matchup { font-size: 0.74rem; color: #94A3B8; margin-bottom: 8px; }
        .p-stats {
            display: flex;
            justify-content: space-around;
            background-color: #0B0E14;
            border-radius: 6px;
            padding: 6px 4px;
            border: 1px solid #1E2636;
        }
        .p-stat-label { font-size: 0.62rem; color: #64748B; text-transform: uppercase; }
        .p-stat-val { font-size: 0.82rem; font-weight: 700; }
        .p-stat-proj { font-size: 0.82rem; font-weight: 800; color: #FF5A00; }

        /* Salary Cap Progress Bar */
        .cap-bar-wrap {
            background-color: #0B0E14;
            border: 1px solid #262F40;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 10px;
        }
        .cap-bar-track {
            width: 100%;
            height: 8px;
            background-color: #1E2636;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 5px;
        }
        .cap-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #FF5A00 0%, #22C55E 100%);
            border-radius: 999px;
        }

        /* Game Environment Card */
        .game-card {
            background: #161B26;
            border: 1px solid #2A3241;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .game-card-prime {
            background: linear-gradient(135deg, #161B26 0%, #2A1508 100%);
            border: 1px solid #FF5A00;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="euro-hero">
            <div class="euro-title">🏀 EUROLEAGUE <span class="euro-accent">PRO DFS SUITE</span></div>
            <div class="euro-subtitle">
                FanTeam 5+2 Daily Optimizer • Correlated Monte Carlo & Field Simulator • Ownership Leverage Engine
            </div>
            <div class="badge-row">
                <span class="euro-badge">Slots: PG • SG • SF • PF • C • 2x FLEX</span>
                <span class="euro-badge">No Captains</span>
                <span class="euro-badge">Contest Tie-Break: Remaining Salary</span>
                <span class="euro-badge">Correlated Sim (P10 / P90 / Top 1%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📂 1. Slate Data")
        uploaded_file = st.file_uploader(
            "Upload Projections (CSV / XLSX)", type=["csv", "xlsx"]
        )

        is_sample_slate = uploaded_file is None
        if not is_sample_slate:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            raw_df = get_sample_data()

        auto_fix_data = st.checkbox(
            "Auto-repair Floor/Ceiling bounds",
            value=True,
            help="Automatically fixes rows where Floor > Proj, Ceiling < Proj, or Floor == 0.",
        )

        st.divider()
        st.markdown("### 💰 2. Budget & Roster Constraints")
        budget = st.number_input(
            "Total Salary Cap (M)",
            min_value=40.0,
            max_value=150.0,
            value=75.0,
            step=0.5,
        )
        min_budget = st.number_input(
            "Minimum Salary Spent (M)",
            min_value=0.0,
            max_value=budget,
            value=round(budget - 3.5, 1),
            step=0.5,
        )
        num_lineups = st.slider("Lineups to Generate", 1, 50, 10)
        min_unique = st.slider("Min Unique Players Between Lineups", 1, 6, 2)
        max_per_team = st.slider("Max Players per Team", 1, 7, 4)
        max_per_game = st.slider("Max Players per Game (Both Teams)", 2, 7, 5)
        global_max_exp = st.slider(
            "Default Max Player Exposure (%)",
            10,
            100,
            100,
            step=5,
            help="Caps how often any non-locked player can appear across your portfolio.",
        )

        st.divider()
        st.markdown("### 🎯 3. Strategy & Blowout Model")
        contest_mode = st.radio(
            "Contest Mode",
            [
                "GPP Tournament (Ceiling / Upside)",
                "Cash Game (Floor / Safety)",
                "Pure Projection",
            ],
        )

        if contest_mode == "Cash Game (Floor / Safety)":
            blend_weight = st.slider("Floor Weight (%)", 0, 100, 50) / 100.0
            min_minutes = st.slider("Min Projected Minutes", 0.0, 35.0, 18.0, 0.5)
            randomness = 0.0
            team_shock_share = 0.0
            env_boost = False
        elif contest_mode == "GPP Tournament (Ceiling / Upside)":
            blend_weight = st.slider("Ceiling Weight (%)", 0, 100, 45) / 100.0
            min_minutes = st.slider("Min Projected Minutes", 0.0, 35.0, 12.0, 0.5)
            randomness = st.slider("Portfolio Randomness (%)", 0, 25, 6) / 100.0
            team_shock_share = (
                st.slider(
                    "Correlated Team Shock Share (%)",
                    0,
                    80,
                    45,
                    help="Applies shared team-level variance so multi-entry lineups rotate stacks naturally.",
                )
                / 100.0
            )
            env_boost = st.checkbox("Boost High Game Totals (+3.5% max)", value=True)
        else:
            blend_weight = 0.0
            min_minutes = st.slider("Min Projected Minutes", 0.0, 35.0, 0.0, 0.5)
            randomness = st.slider("Portfolio Randomness (%)", 0, 25, 0) / 100.0
            team_shock_share = 0.40
            env_boost = False

        with st.expander("💨 Blowout Adjustment Settings"):
            blowout_threshold = st.slider(
                "Favored Spread Threshold (Pts)", 4.0, 20.0, 10.0, 0.5
            )
            blowout_penalty_pct = st.slider(
                "Starter (24+ Min) Penalty (%)", 0.0, 20.0, 0.0, 0.5
            )

        with st.expander("📊 Ownership & Leverage Settings"):
            use_estimated_own = st.checkbox(
                "Estimate Ownership (700% Softmax)",
                value=True,
                help="Estimates ownership from Value, Proj, Min, and Win_Prob when CSV lacks Own%.",
            )
            own_sharpness = st.slider(
                "Ownership Chalk Concentration", 0.5, 3.0, 1.6, 0.1
            )
            if contest_mode == "GPP Tournament (Ceiling / Upside)":
                own_penalty_lambda = st.slider(
                    "GPP Ownership Penalty (λ)",
                    0.0,
                    0.30,
                    0.05,
                    0.01,
                    help="Subtracts λ × Own% from Opt_Score to create tournament leverage.",
                )
                max_lineup_own = float(
                    st.slider("Max Total Lineup Ownership (%)", 100, 400, 260, 5)
                )
                chalk_threshold = float(
                    st.slider("Chalk Player Threshold (%)", 15, 60, 30, 5)
                )
                max_chalk_count = st.slider(
                    f"Max Players Over {int(chalk_threshold)}% Owned", 1, 7, 3
                )
            else:
                own_penalty_lambda = 0.0
                max_lineup_own = 700.0
                chalk_threshold = 50.0
                max_chalk_count = 7

        with st.expander("🎲 Monte Carlo & Solver Settings"):
            num_sims = st.select_slider(
                "Monte Carlo Trials", options=[1000, 2500, 5000, 10000], value=2500
            )
            target_score = st.number_input(
                "Target Contest Score (Pts)", 100.0, 300.0, 175.0, 2.5
            )
            solver_time_limit = st.number_input(
                "Solver Time Limit per Lineup (s)", 1.0, 30.0, 5.0, 1.0
            )

    # ---------------------------------------------------------
    # WARNING FOR DEMO DATA & DATA VALIDATION
    # ---------------------------------------------------------
    if is_sample_slate:
        st.warning(
            "⚠️ **DEMO / PLACEHOLDER SLATE ACTIVE:** You are viewing built-in sample numbers. "
            "Upload today's FanTeam projections CSV in the sidebar before building real contest lineups!"
        )

    df = normalize_dataframe(raw_df)
    df["Max_Exp%"] = float(global_max_exp)

    val_report = validate_slate_data(df)
    if val_report["total_issues"] > 0:
        with st.expander(
            f"🔍 Slate Data Validation Alert ({val_report['total_issues']} Anomalies Detected)",
            expanded=not auto_fix_data,
        ):
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Floor > Proj", len(val_report["floor_gt_proj"]))
            v2.metric("Ceiling < Proj", len(val_report["ceil_lt_proj"]))
            v3.metric("Floor <= 0", len(val_report["zero_floor"]))
            v4.metric("Team Min > 200", len(val_report["high_team_mins"]))
            if not val_report["floor_gt_proj"].empty:
                st.caption("Players with Floor > Proj:")
                st.dataframe(
                    val_report["floor_gt_proj"], width="stretch", hide_index=True
                )
            if not val_report["ceil_lt_proj"].empty:
                st.caption("Players with Ceiling < Proj:")
                st.dataframe(
                    val_report["ceil_lt_proj"], width="stretch", hide_index=True
                )
            if not val_report["zero_floor"].empty:
                st.caption("Players with Floor <= 0:")
                st.dataframe(val_report["zero_floor"], width="stretch", hide_index=True)
            if not val_report["high_team_mins"].empty:
                st.caption("Teams with Total Listed Minutes > 200:")
                st.dataframe(
                    val_report["high_team_mins"], width="stretch", hide_index=True
                )

    if auto_fix_data:
        df = repair_slate_anomalies(df)

    # ---------------------------------------------------------
    # TOP-LEVEL NAVIGATION TABS
    # ---------------------------------------------------------
    nav_opt, nav_slate, nav_cal = st.tabs(
        [
            "🛠️ Lineup Optimizer & Simulator",
            "🏟️ Slate Overview & Game Environments",
            "📈 Post-Contest Ownership Calibration",
        ]
    )

    # ---------------------------------------------------------
    # TAB 2: SLATE OVERVIEW & GAME ENVIRONMENTS
    # ---------------------------------------------------------
    with nav_slate:
        st.subheader("🏟️ EuroLeague Matchup Environments")
        games_summary = []
        for g_key, g_df in df.groupby("Game_Key"):
            teams_in_game = sorted(g_df["TEAM"].unique().tolist())
            t1 = teams_in_game[0]
            t2 = teams_in_game[1] if len(teams_in_game) > 1 else g_df["Opp"].iloc[0]
            t1_pts = g_df[g_df["TEAM"] == t1]["Team_Pts"].median()
            t2_pts = (
                g_df[g_df["TEAM"] == t2]["Team_Pts"].median()
                if t2 in g_df["TEAM"].values
                else g_df["Opp_Pts"].median()
            )
            pace = g_df["Pace"].median()
            t1_wp = g_df[g_df["TEAM"] == t1]["Win_Prob"].median()
            games_summary.append(
                {
                    "Game_Key": g_key,
                    "Team_A": t1,
                    "Team_B": t2,
                    "Team_A_Pts": round(float(t1_pts), 1),
                    "Team_B_Pts": round(float(t2_pts), 1),
                    "Total_Pts": round(float(t1_pts + t2_pts), 1),
                    "Spread": round(abs(float(t1_pts - t2_pts)), 1),
                    "Favorite": t1 if t1_pts >= t2_pts else t2,
                    "Pace": round(float(pace), 1),
                    "Fav_Win_Prob": round(float(max(t1_wp, 1.0 - t1_wp) * 100.0), 1),
                }
            )
        games_df = pd.DataFrame(games_summary).sort_values(
            by="Total_Pts", ascending=False
        )

        g_cols = st.columns(min(3, max(1, len(games_df))))
        for idx, (_, g_row) in enumerate(games_df.iterrows()):
            is_prime = idx == 0
            card_cls = "game-card-prime" if is_prime else "game-card"
            prime_badge = (
                '<span class="euro-badge" style="float:right;">🔥 PRIME ENVIRONMENT</span>'
                if is_prime
                else ""
            )
            with g_cols[idx % len(g_cols)]:
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        {prime_badge}
                        <div style="font-size:1.15rem; font-weight:800; margin-bottom:6px;">
                            {g_row["Team_A"]} ({g_row["Team_A_Pts"]}) vs {g_row["Team_B"]} ({g_row["Team_B_Pts"]})
                        </div>
                        <div style="color:#94A3B8; font-size:0.85rem; margin-bottom:10px;">
                            Favorite: <b>{g_row["Favorite"]} (-{g_row["Spread"]})</b> • Win Prob: <b>{g_row["Fav_Win_Prob"]}%</b>
                        </div>
                        <div class="p-stats">
                            <div>
                                <div class="p-stat-label">O/U TOTAL</div>
                                <div class="p-stat-proj">{g_row["Total_Pts"]} Pts</div>
                            </div>
                            <div>
                                <div class="p-stat-label">PACE</div>
                                <div class="p-stat-val">{g_row["Pace"]}</div>
                            </div>
                            <div>
                                <div class="p-stat-label">SPREAD</div>
                                <div class="p-stat-val">{g_row["Spread"]} Pts</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---------------------------------------------------------
    # TAB 3: POST-CONTEST OWNERSHIP CALIBRATION
    # ---------------------------------------------------------
    with nav_cal:
        st.subheader("📈 Calibrate Ownership Model Against Actual Contest Logs")
        st.caption(
            "Upload a past slate CSV containing both projections and actual `Real_Own%` to inspect model accuracy and find the optimal sharpness parameter."
        )
        cal_file = st.file_uploader(
            "Upload Historical Slate with `Real_Own%`", type=["csv"], key="cal_upload"
        )
        if cal_file is not None:
            cal_df = normalize_dataframe(pd.read_csv(cal_file))
            if "Real_Own%" in cal_df.columns:
                best_s, best_mae = 1.6, 999.0
                for s_try in np.linspace(0.5, 3.0, 26):
                    pred = estimate_ownership(cal_df, sharpness=float(s_try))
                    mae = float(np.abs(pred - cal_df["Real_Own%"]).mean())
                    if mae < best_mae:
                        best_mae = mae
                        best_s = round(float(s_try), 2)
                st.success(
                    f"Optimal Ownership Sharpness: **{best_s}** (Mean Absolute Error: {best_mae:.2f}%)"
                )
            else:
                st.info("Add a `Real_Own%` column to your CSV to fit sharpness.")

    # ---------------------------------------------------------
    # TAB 1: MAIN OPTIMIZER & SIMULATOR WORKSPACE
    # ---------------------------------------------------------
    with nav_opt:
        all_teams = sorted(df["TEAM"].unique().tolist())
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 🏢 Team Filter & Late-Swap")
            included_teams = st.multiselect(
                "Include ONLY Teams (Blank = All)", options=all_teams
            )
            excluded_teams = st.multiselect(
                "Exclude Teams / Tipped-Off Games",
                options=[t for t in all_teams if t not in included_teams],
            )

        team_filtered_df = df.copy()
        if included_teams:
            team_filtered_df = team_filtered_df[
                team_filtered_df["TEAM"].isin(included_teams)
            ]
        if excluded_teams:
            team_filtered_df = team_filtered_df[
                ~team_filtered_df["TEAM"].isin(excluded_teams)
            ]

        before_min_count = len(team_filtered_df)
        min_filtered_df = team_filtered_df[
            team_filtered_df["Min"] >= min_minutes
        ].copy()
        hidden_by_min = before_min_count - len(min_filtered_df)

        available_players = sorted(min_filtered_df["Name"].unique().tolist())
        all_player_names = sorted(df["Name"].unique().tolist())
        active_teams = sorted(min_filtered_df["TEAM"].unique().tolist())

        with c2:
            st.markdown("##### 🔒 Player Locks, Late-Swap & Fades")
            locked_players = st.multiselect(
                "Lock / Freeze Players (Max 7)",
                options=all_player_names,
                max_selections=7,
                help="Locked players bypass the min-minutes filter and are forced into every lineup.",
            )
            excluded_players = st.multiselect(
                "Exclude (Fade) Players",
                options=[p for p in available_players if p not in locked_players],
            )

        with c3:
            st.markdown("##### 🔥 Stacking & Bring-Back Rules")
            if max_per_team < 2:
                st.info(
                    "Increase 'Max Players per Team' to at least 2 in the sidebar to enable stacking."
                )
                stack_teams = []
                stack_rules = {}
                bring_back_enabled = False
                auto_bring_back_min = 0
            else:
                stack_teams = st.multiselect(
                    "Force Team Stack(s)", options=active_teams
                )
                stack_rules = {}
                for t in stack_teams:
                    stack_rules[t] = st.number_input(
                        f"Min players from {t}",
                        min_value=2,
                        max_value=max(2, max_per_team),
                        value=min(3, max(2, max_per_team)),
                        key=f"stack_{t}",
                    )
                bring_back_enabled = st.checkbox(
                    "GPP Bring-Back Rule (Require ≥1 Opponent Player)",
                    value=(contest_mode == "GPP Tournament (Ceiling / Upside)"),
                    help="Whenever a team is stacked (≥3 players or forced stack), requires at least 1 player from their opponent.",
                )
                auto_bring_back_min = 3 if bring_back_enabled else 0

        with st.expander("🔗 Advanced Correlation Rules (Force-Pairs & Ban-Pairs)"):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.caption(
                    "Force-Pair: Select pairs of players that must always be rostered together."
                )
                fp_selection = st.multiselect(
                    "Select 2 Players to Force-Pair",
                    options=all_player_names,
                    max_selections=2,
                    key="fp_sel",
                )
                force_pairs = (
                    [(fp_selection[0], fp_selection[1])]
                    if len(fp_selection) == 2
                    else []
                )
            with p_col2:
                st.caption(
                    "Ban-Pair: Select 2 players that can never appear in the same lineup."
                )
                bp_selection = st.multiselect(
                    "Select 2 Players to Ban Together",
                    options=all_player_names,
                    max_selections=2,
                    key="bp_sel",
                )
                ban_pairs = (
                    [(bp_selection[0], bp_selection[1])]
                    if len(bp_selection) == 2
                    else []
                )

        pre_edit_pool = (
            df[
                (
                    df["Name"].isin(min_filtered_df["Name"])
                    | df["Name"].isin(locked_players)
                )
                & (~df["Name"].isin(excluded_players))
            ]
            .copy()
            .reset_index(drop=True)
        )

        pre_edit_pool = compute_metrics_after_edit(
            df=pre_edit_pool,
            contest_mode=contest_mode,
            blend_weight=blend_weight,
            env_boost=env_boost,
            blowout_threshold=blowout_threshold,
            blowout_penalty_pct=blowout_penalty_pct,
            own_penalty_lambda=own_penalty_lambda,
            use_estimated_own=use_estimated_own,
            own_sharpness=own_sharpness,
        )

        if hidden_by_min > 0:
            st.caption(
                f"ℹ️ **{hidden_by_min} player(s) hidden** by the `Min >= {min_minutes:.1f}` filter "
                f"({len(pre_edit_pool)} active players remaining in pool)."
            )

        with st.expander(
            f"📋 Edit Active Player Pool ({len(pre_edit_pool)} Players) — Edits Drive Solver Live",
            expanded=False,
        ):
            st.caption(
                "Edit `Salary`, `Proj`, `Floor`, `Ceiling`, `Own%`, or `Max_Exp%` below. "
                "`Opt_Score` and `Value` automatically recompute from your edits!"
            )
            editor_cols = [
                "Name",
                "Position",
                "TEAM",
                "Opp",
                "Salary",
                "Proj",
                "Floor",
                "Ceiling",
                "Own%",
                "Max_Exp%",
                "Min",
                "Value",
                "Favored_By",
                "Opt_Score",
                "Game_Key",
                "Game_Total",
                "Pace",
                "Win_Prob",
            ]
            editor_cols = [c for c in editor_cols if c in pre_edit_pool.columns]
            raw_edited_pool = st.data_editor(
                pre_edit_pool[editor_cols],
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Salary": st.column_config.NumberColumn("Salary", format="%.1fM"),
                    "Proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                    "Floor": st.column_config.NumberColumn(
                        "Floor (P10)", format="%.1f"
                    ),
                    "Ceiling": st.column_config.NumberColumn(
                        "Ceiling (P90)", format="%.1f"
                    ),
                    "Own%": st.column_config.ProgressColumn(
                        "Own%", format="%.1f%%", min_value=0.0, max_value=80.0
                    ),
                    "Max_Exp%": st.column_config.NumberColumn(
                        "Max Exp%", format="%.0f%%", min_value=0.0, max_value=100.0
                    ),
                    "Opt_Score": st.column_config.ProgressColumn(
                        "Opt Score",
                        format="%.1f",
                        min_value=0.0,
                        max_value=float(
                            max(pre_edit_pool["Opt_Score"].max(), 40.0)
                            if len(pre_edit_pool) > 0
                            else 40.0
                        ),
                    ),
                },
            )

        edited_pool = compute_metrics_after_edit(
            df=raw_edited_pool,
            contest_mode=contest_mode,
            blend_weight=blend_weight,
            env_boost=env_boost,
            blowout_threshold=blowout_threshold,
            blowout_penalty_pct=blowout_penalty_pct,
            own_penalty_lambda=own_penalty_lambda,
            use_estimated_own=False,
            own_sharpness=own_sharpness,
        )

        # ---------------------------------------------------------
        # SOLVE & SAVE IN SESSION STATE
        # ---------------------------------------------------------
        if st.button(
            "🚀 GENERATE & SIMULATE OPTIMAL LINEUPS",
            type="primary",
            width="stretch",
        ):
            missing_pos = [
                p
                for p in REQUIRED_POSITIONS
                if p not in edited_pool["Position"].unique()
            ]
            if missing_pos:
                st.error(
                    f"Cannot build a 5+2 lineup! Filtered pool is missing position(s): {', '.join(missing_pos)}"
                )
            elif len(edited_pool) < 7:
                st.error(
                    "Not enough players in the active pool to build a 7-player squad."
                )
            else:
                with st.spinner(
                    "Solving MILP portfolio & running correlated Monte Carlo field simulations..."
                ):
                    lineups = generate_lineups(
                        data=edited_pool,
                        num_lineups=num_lineups,
                        budget=budget,
                        min_budget=min_budget,
                        locked=locked_players,
                        stacks=stack_rules,
                        bring_back_enabled=bring_back_enabled,
                        auto_bring_back_min=auto_bring_back_min,
                        max_team=max_per_team,
                        max_per_game=max_per_game,
                        min_unique=min_unique,
                        randomness=randomness,
                        team_shock_share=team_shock_share,
                        max_lineup_own=max_lineup_own,
                        max_chalk_count=max_chalk_count,
                        chalk_threshold=chalk_threshold,
                        force_pairs=force_pairs,
                        ban_pairs=ban_pairs,
                        solver_time_limit=solver_time_limit,
                    )

                    if not lineups:
                        st.session_state["opt_results"] = None
                        st.warning(
                            "No feasible lineups found! Try relaxing your Ownership cap, Team Stacks, or Minimum Salary."
                        )
                    else:
                        sim_df = run_monte_carlo_and_field_sim(
                            pool_df=edited_pool,
                            lineups=lineups,
                            budget=budget,
                            target_score=target_score,
                            num_sims=num_sims,
                        )
                        compact_df = build_compact_export(
                            lineups=lineups, sim_df=sim_df, budget=budget
                        )
                        detailed_rows = []
                        for idx, l_df in enumerate(lineups):
                            temp = l_df.copy()
                            temp.insert(0, "Lineup_#", idx + 1)
                            detailed_rows.append(temp)
                        detailed_df = pd.concat(detailed_rows, ignore_index=True)

                        st.session_state["opt_results"] = {
                            "lineups": lineups,
                            "sim_df": sim_df,
                            "compact_df": compact_df,
                            "detailed_df": detailed_df,
                            "budget": budget,
                            "contest_mode": contest_mode,
                            "target_score": target_score,
                        }

        # ---------------------------------------------------------
        # RENDER PERSISTENT RESULTS FROM SESSION STATE
        # ---------------------------------------------------------
        saved = st.session_state.get("opt_results")
        if saved:
            lineups = saved["lineups"]
            sim_df = saved["sim_df"]
            compact_df = saved["compact_df"]
            detailed_df = saved["detailed_df"]
            saved_budget = saved["budget"]

            st.divider()
            dl1, dl2, dl3 = st.columns([2, 1, 1])
            with dl1:
                st.success(
                    f"✅ Active Portfolio: **{len(lineups)} Squad(s)** ({saved['contest_mode']})"
                )
            with dl2:
                _safe_download_button(
                    label="📥 Export Compact (1 Row/Lineup)",
                    data=compact_df.to_csv(index=False).encode("utf-8"),
                    file_name="fanteam_euroleague_compact.csv",
                    key="dl_compact",
                )
            with dl3:
                _safe_download_button(
                    label="📥 Export Full Player Rows",
                    data=detailed_df.to_csv(index=False).encode("utf-8"),
                    file_name="fanteam_euroleague_detailed.csv",
                    key="dl_detailed",
                )

            res_tab1, res_tab2, res_tab3 = st.tabs(
                [
                    "🏀 Visual Court Cards & Sim Metrics",
                    "⚡ Quick-Entry Compact Table & Field Sim",
                    "📊 Portfolio Exposure & Stack Analytics",
                ]
            )

            with res_tab1:
                for i, l_df in enumerate(lineups):
                    tot_sal = float(l_df["Salary"].sum())
                    rem_sal = saved_budget - tot_sal
                    cap_pct = min(100.0, (tot_sal / saved_budget) * 100.0)
                    tot_proj = float(l_df["Proj"].sum())
                    tot_own = float(l_df["Own%"].sum())
                    s_row = sim_df.iloc[i]

                    team_counts = l_df["TEAM"].value_counts()
                    stack_str = ", ".join(
                        [f"{k} ({v})" for k, v in team_counts.items() if v >= 2]
                    )

                    st.markdown(
                        f"#### 🏆 Squad #{i + 1} — Stacks: `{stack_str if stack_str else 'Balanced'}`"
                    )

                    st.markdown(
                        f"""
                        <div class="cap-bar-wrap">
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                                <span><b>Salary Used:</b> {tot_sal:.1f}M / {saved_budget:.1f}M ({cap_pct:.1f}%)</span>
                                <span style="color:#22C55E;"><b>Contest Tie-Break Rem:</b> {rem_sal:.1f}M</span>
                            </div>
                            <div class="cap-bar-track">
                                <div class="cap-bar-fill" style="width:{cap_pct:.1f}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Proj Pts", f"{tot_proj:.1f}")
                    m2.metric("Sim Mean", f"{s_row['Sim_Mean']:.1f}")
                    m3.metric(
                        "Sim P10 / P90",
                        f"{s_row['Sim_P10']:.1f} / {s_row['Sim_P90']:.1f}",
                    )
                    m4.metric("Cumulative Own%", f"{tot_own:.1f}%")
                    m5.metric("Cash Rate (Top 17%)", f"{s_row['Cash_Top17%']:.1f}%")
                    m6.metric("GPP Top-1% Rate", f"{s_row['GPP_Top1%']:.1f}%")

                    cards_html = ['<div class="lineup-grid">']
                    for _, p_row in l_df.iterrows():
                        slot_str = str(p_row["Slot"])
                        pill_cls = (
                            "slot-pill-flex"
                            if slot_str.startswith("FLEX")
                            else "slot-pill"
                        )
                        t_code = str(p_row["TEAM"])
                        bg_col, txt_col = TEAM_COLORS.get(
                            t_code, ("#334155", "#FFFFFF")
                        )
                        cards_html.append(
                            f"""
                            <div class="court-card">
                                <div class="slot-header">
                                    <span class="{pill_cls}">{slot_str}</span>
                                    <span class="team-chip" style="background:{bg_col}; color:{txt_col};">{t_code}</span>
                                </div>
                                <div class="p-name" title="{p_row["Name"]}">{p_row["Name"]}</div>
                                <div class="p-matchup">{p_row["Position"]} • vs {p_row["Opp"]} ({p_row["Own%"]:.1f}% Own)</div>
                                <div class="p-stats">
                                    <div>
                                        <div class="p-stat-label">SAL</div>
                                        <div class="p-stat-val">{p_row["Salary"]:.1f}M</div>
                                    </div>
                                    <div>
                                        <div class="p-stat-label">PROJ</div>
                                        <div class="p-stat-proj">{p_row["Proj"]:.1f}</div>
                                    </div>
                                    <div>
                                        <div class="p-stat-label">P10-P90</div>
                                        <div class="p-stat-val">{p_row["Floor"]:.0f}-{p_row["Ceiling"]:.0f}</div>
                                    </div>
                                </div>
                            </div>
                            """
                        )
                    cards_html.append("</div>")
                    st.markdown("".join(cards_html), unsafe_allow_html=True)

            with res_tab2:
                st.subheader("⚡ Compact FanTeam Entry Table & Field Simulation")
                st.dataframe(compact_df, width="stretch", hide_index=True)

            with res_tab3:
                e1, e2 = st.columns([2, 1])
                with e1:
                    st.subheader("Player Exposure vs. Slate Ownership")
                    exp_df = (
                        detailed_df.groupby(
                            ["Name", "Position", "TEAM", "Salary", "Proj", "Own%"]
                        )
                        .size()
                        .reset_index(name="Squads")
                    )
                    exp_df["Portfolio_%"] = (
                        (exp_df["Squads"] / len(lineups)) * 100.0
                    ).round(1)
                    exp_df["Leverage_%"] = (
                        exp_df["Portfolio_%"] - exp_df["Own%"]
                    ).round(1)
                    exp_df = exp_df.sort_values(
                        by=["Portfolio_%", "Proj"], ascending=[False, False]
                    )
                    st.dataframe(
                        exp_df,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Portfolio_%": st.column_config.ProgressColumn(
                                "Portfolio %",
                                format="%.1f%%",
                                min_value=0.0,
                                max_value=100.0,
                            ),
                            "Own%": st.column_config.NumberColumn(
                                "Field Own%", format="%.1f%%"
                            ),
                            "Leverage_%": st.column_config.NumberColumn(
                                "Leverage (+/-)", format="%+.1f%%"
                            ),
                        },
                    )
                with e2:
                    st.subheader("Team Stack Exposure")
                    t_exp = (
                        detailed_df.groupby("TEAM")
                        .size()
                        .reset_index(name="Total_Spots")
                    )
                    t_exp["Avg_Per_Lineup"] = (
                        t_exp["Total_Spots"] / len(lineups)
                    ).round(2)
                    st.dataframe(
                        t_exp.sort_values(by="Total_Spots", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )


if __name__ == "__main__":
    main()
