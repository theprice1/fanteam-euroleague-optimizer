import io

import numpy as np
import pandas as pd
import pulp
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="EuroLeague Fantasy Optimizer | FanTeam",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# EUROLEAGUE CUSTOM CSS STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Hero Banner */
    .euro-hero {
        background: linear-gradient(135deg, #161B26 0%, #0B0E14 60%, #2A1205 100%);
        border: 1px solid #2A3241;
        border-left: 6px solid #FF5A00;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    }
    .euro-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .euro-accent {
        color: #FF5A00;
    }
    .euro-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 14px;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .euro-badge {
        background-color: rgba(255, 90, 0, 0.14);
        border: 1px solid rgba(255, 90, 0, 0.4);
        color: #FF8533;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section Filter Cards */
    .filter-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        border-bottom: 2px solid #FF5A00;
        padding-bottom: 6px;
        margin-bottom: 12px;
        display: inline-block;
    }

    /* Lineup Header Banner */
    .lineup-banner {
        background: linear-gradient(90deg, #161B26 0%, #1F2737 100%);
        border: 1px solid #2D3748;
        border-top: 3px solid #FF5A00;
        border-radius: 10px;
        padding: 14px 20px;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* 7-Player Visual Court Cards */
    .player-card {
        background: linear-gradient(180deg, #181E2C 0%, #10141D 100%);
        border: 1px solid #2A3241;
        border-radius: 10px;
        padding: 12px 10px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        height: 100%;
    }
    .pos-pill {
        display: inline-block;
        background-color: #FF5A00;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.72rem;
        padding: 2px 9px;
        border-radius: 6px;
        margin-bottom: 6px;
        letter-spacing: 0.6px;
    }
    .p-name {
        font-weight: 700;
        font-size: 0.92rem;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 3px;
    }
    .p-matchup {
        font-size: 0.76rem;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .p-stats {
        display: flex;
        justify-content: space-around;
        background-color: #0B0E14;
        border-radius: 6px;
        padding: 6px 4px;
        border: 1px solid #1E2636;
    }
    .p-stat-box {
        line-height: 1.15;
    }
    .p-stat-label {
        font-size: 0.65rem;
        color: #64748B;
        text-transform: uppercase;
    }
    .p-stat-val {
        font-size: 0.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .p-stat-proj {
        font-size: 0.85rem;
        font-weight: 800;
        color: #FF5A00;
    }

    /* Metric Styling Override */
    div[data-testid="stMetric"] {
        background-color: #161B26;
        border: 1px solid #262F40;
        border-radius: 8px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="euro-hero">
        <div class="euro-title">🏀 EUROLEAGUE <span class="euro-accent">DFS OPTIMIZER</span></div>
        <div class="euro-subtitle">
            Built for FanTeam Daily Basketball Tournaments • Linear Programming Engine
        </div>
        <div class="badge-row">
            <span class="euro-badge">5+2 Format (7 Players)</span>
            <span class="euro-badge">Min 1 PG • SG • SF • PF • C</span>
            <span class="euro-badge">No Captains</span>
            <span class="euro-badge">Tie-Breaker: Highest Remaining Budget</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SAMPLE DATA GENERATOR (Matches Your Exact Schema)
# ---------------------------------------------------------
@st.cache_data
def get_sample_data():
    csv_data = """Name,position,TEAM,Opp,Salary,Proj,Floor,Ceiling,Value,Min,Team_Pts,Pace,Win_Prob
Nadir Hifi,SG,PBB,VIR,14.2,28.2,24.1,31.9,2.0,25.8,90.5,74.4,0.577
Elijah Bryant,SG,HTA,BJK,15.6,28.0,21.8,33.7,1.8,33.1,91.5,72.2,0.724
TJ Shorts,PG,PBB,VIR,15.8,31.4,26.0,37.5,2.0,29.5,90.5,74.4,0.577
Sasha Vezenkov,PF,OLY,FBB,16.5,33.0,27.5,39.2,2.0,31.0,86.0,70.1,0.640
Mathias Lessort,C,PAO,RMD,15.2,29.5,24.0,35.8,1.9,28.5,88.0,73.0,0.550
Kendrick Nunn,PG,PAO,RMD,14.8,27.8,20.5,36.0,1.9,30.2,88.0,73.0,0.550
Facundo Campazzo,PG,RMD,PAO,13.9,26.4,21.2,32.1,1.9,27.8,86.5,73.0,0.450
Mario Hezonja,SF,RMD,PAO,13.4,25.1,18.9,33.4,1.9,28.0,86.5,73.0,0.450
Tornike Shengelia,PF,VIR,PBB,13.8,26.0,21.5,31.0,1.9,27.5,87.0,74.4,0.423
Will Clyburn,SF,VIR,PBB,12.1,22.4,17.0,28.5,1.85,26.0,87.0,74.4,0.423
Tyson Ward,SF,PBB,VIR,10.5,20.8,16.2,26.4,1.98,26.5,90.5,74.4,0.577
Mikael Jantunen,PF,PBB,VIR,9.2,18.5,14.8,23.1,2.01,25.0,90.5,74.4,0.577
Kevarrius Hayes,C,PBB,VIR,8.4,16.9,13.0,21.5,2.01,22.0,90.5,74.4,0.577
Jonathan Motley,C,HTA,BJK,13.2,25.6,19.8,31.4,1.94,26.4,91.5,72.2,0.724
Patrick Beverley,PG,HTA,BJK,11.0,21.5,16.5,27.2,1.95,25.5,91.5,72.2,0.724
Nigel Hayes-Davis,SF,FBB,OLY,14.0,26.8,22.1,32.5,1.91,33.5,82.5,70.1,0.360
Moses Wright,C,OLY,FBB,8.8,17.5,12.5,23.0,1.99,20.5,86.0,70.1,0.640
Thomas Walkup,PG,OLY,FBB,8.5,16.8,13.5,20.5,1.98,24.0,86.0,70.1,0.640
Isaac Bonga,SF,PAR,BAY,7.6,15.2,11.8,19.5,2.00,23.5,84.0,71.5,0.580
Aleksej Pokusevski,PF,PAR,BAY,7.2,14.5,10.2,20.1,2.01,19.5,84.0,71.5,0.580
"""
    return pd.read_csv(io.StringIO(csv_data.strip()))


# ---------------------------------------------------------
# SIDEBAR: DATA UPLOAD & SETTINGS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 1. Slate Projections")
    uploaded_file = st.file_uploader(
        "Upload Daily Projections (CSV / XLSX)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"Loaded: {uploaded_file.name}")
    else:
        st.info("Using built-in EuroLeague sample slate. Upload a file to override.")
        df = get_sample_data()

    st.divider()
    st.markdown("### 💰 2. Budget & Lineup Rules")

    budget = st.number_input(
        "Total Salary Cap (M)",
        min_value=40.0,
        max_value=150.0,
        value=75.0,
        step=0.5,
        help="FanTeam daily budget (default 75M).",
    )

    min_budget = st.number_input(
        "Minimum Salary Spent (M)",
        min_value=0.0,
        max_value=budget,
        value=round(budget - 3.0, 1),
        step=0.5,
    )

    num_lineups = st.slider("Lineups to Generate", 1, 30, 5)
    min_unique = st.slider("Min Unique Players Between Lineups", 1, 6, 2)
    max_per_team = st.slider("Max Players per Real-Life Team", 1, 7, 4)

    st.divider()
    st.markdown("### 🎯 3. Contest Strategy Mode")
    contest_mode = st.radio(
        "Optimization Mode",
        [
            "Cash Game (Floor / Safety)",
            "GPP Tournament (Ceiling / Upside)",
            "Pure Projection",
        ],
    )

    if contest_mode == "Cash Game (Floor / Safety)":
        floor_weight = (
            st.slider(
                "Floor Weight (%)",
                0,
                100,
                50,
                help="Blends Proj and Floor to prioritize high-minute safe plays.",
            )
            / 100.0
        )
        min_minutes = st.slider("Min Projected Minutes", 0.0, 35.0, 18.0, 0.5)
        randomness = 0.0
        env_boost = False
    elif contest_mode == "GPP Tournament (Ceiling / Upside)":
        ceil_weight = (
            st.slider(
                "Ceiling Weight (%)",
                0,
                100,
                45,
                help="Blends Proj and Ceiling to hunt tournament-winning upside.",
            )
            / 100.0
        )
        min_minutes = st.slider("Min Projected Minutes", 0.0, 35.0, 12.0, 0.5)
        randomness = (
            st.slider(
                "Multi-Entry Randomness (%)",
                0,
                25,
                5,
                help="Adds slight variance across lineups 2..N.",
            )
            / 100.0
        )
        env_boost = st.checkbox("Vegas Environment Boost (Team Pts / Pace)", value=True)
    else:
        min_minutes = st.slider("Min Projected Minutes", 0.0, 35.0, 0.0, 0.5)
        randomness = st.slider("Multi-Entry Randomness (%)", 0, 25, 0) / 100.0
        env_boost = False

# ---------------------------------------------------------
# DATA NORMALIZATION & SCORE CALCULATION
# ---------------------------------------------------------
df.columns = [c.strip() for c in df.columns]
col_map = {
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
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

df["Position"] = df["Position"].astype(str).str.upper().str.strip()
df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip()

if contest_mode == "Cash Game (Floor / Safety)":
    df["Opt_Score"] = ((1.0 - floor_weight) * df["Proj"]) + (floor_weight * df["Floor"])
elif contest_mode == "GPP Tournament (Ceiling / Upside)":
    df["Opt_Score"] = ((1.0 - ceil_weight) * df["Proj"]) + (ceil_weight * df["Ceiling"])
    if env_boost and "Team_Pts" in df.columns and df["Team_Pts"].max() > 0:
        pts_norm = (df["Team_Pts"] - df["Team_Pts"].min()) / (
            df["Team_Pts"].max() - df["Team_Pts"].min() + 1e-6
        )
        df["Opt_Score"] = df["Opt_Score"] * (1.0 + 0.03 * pts_norm)
else:
    df["Opt_Score"] = df["Proj"]

df["Opt_Score"] = df["Opt_Score"].round(2)

# ---------------------------------------------------------
# MAIN CONTROLS: TEAMS, PLAYERS & STACKING
# ---------------------------------------------------------
all_teams = sorted(df["TEAM"].unique().tolist())

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="filter-header">🏢 Team Filter (Include / Exclude)</div>',
        unsafe_allow_html=True,
    )
    included_teams = st.multiselect(
        "Include ONLY These Teams (Blank = All)",
        options=all_teams,
        default=[],
    )
    excluded_teams = st.multiselect(
        "Exclude Teams from Pool",
        options=[t for t in all_teams if t not in included_teams],
        default=[],
    )

available_df = df.copy()
if included_teams:
    available_df = available_df[available_df["TEAM"].isin(included_teams)]
if excluded_teams:
    available_df = available_df[~available_df["TEAM"].isin(excluded_teams)]
if "Min" in available_df.columns:
    available_df = available_df[available_df["Min"] >= min_minutes]

available_players = sorted(available_df["Name"].unique().tolist())
active_teams = sorted(available_df["TEAM"].unique().tolist())

with col2:
    st.markdown(
        '<div class="filter-header">🔒 Player Locks & Fades</div>',
        unsafe_allow_html=True,
    )
    locked_players = st.multiselect(
        "Lock Players into Every Lineup (Max 7)",
        options=available_players,
        default=[],
        max_selections=7,
    )
    excluded_players = st.multiselect(
        "Exclude (Fade) Players",
        options=[p for p in available_players if p not in locked_players],
        default=[],
    )

with col3:
    st.markdown(
        '<div class="filter-header">🔥 Team Stacking Builder</div>',
        unsafe_allow_html=True,
    )
    stack_teams = st.multiselect(
        "Select Team(s) to Force Stack",
        options=active_teams,
        help="Forces the solver to roster at least X players from selected teams.",
    )
    stack_rules = {}
    for t in stack_teams:
        stack_rules[t] = st.number_input(
            f"Min players from {t}",
            min_value=2,
            max_value=max_per_team,
            value=min(3, max_per_team),
            key=f"stack_{t}",
        )

pool_df = (
    df[
        (df["Name"].isin(available_df["Name"]) | df["Name"].isin(locked_players))
        & (~df["Name"].isin(excluded_players))
    ]
    .copy()
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# SLATE SUMMARY METRICS & INTERACTIVE POOL
# ---------------------------------------------------------
s1, s2, s3, s4 = st.columns(4)
s1.metric("Available Player Pool", f"{len(pool_df)} Players")
s2.metric("Active EuroLeague Teams", f"{pool_df['TEAM'].nunique()} Teams")
s3.metric(
    "Locked / Excluded Players", f"{len(locked_players)} / {len(excluded_players)}"
)
s4.metric("Active Strategy Mode", contest_mode.split(" (")[0])

with st.expander(
    f"📋 Inspect & Edit Active Player Pool ({len(pool_df)} Players)", expanded=False
):
    st.caption(
        "Double-click any cell to tweak `Salary`, `Proj`, or `Opt_Score` on the fly before running the solver."
    )
    edited_pool = st.data_editor(
        pool_df,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Salary": st.column_config.NumberColumn("Salary (M)", format="%.1fM"),
            "Proj": st.column_config.NumberColumn("Proj Pts", format="%.1f"),
            "Floor": st.column_config.NumberColumn("Floor", format="%.1f"),
            "Ceiling": st.column_config.NumberColumn("Ceiling", format="%.1f"),
            "Opt_Score": st.column_config.ProgressColumn(
                "Opt Score",
                format="%.1f",
                min_value=0,
                max_value=float(max(pool_df["Opt_Score"].max(), 40.0))
                if len(pool_df) > 0
                else 40.0,
            ),
            "Win_Prob": st.column_config.ProgressColumn(
                "Win Prob", format="%.3f", min_value=0.0, max_value=1.0
            ),
        },
    )


# ---------------------------------------------------------
# SOLVER ENGINE (PuLP Linear Programming)
# ---------------------------------------------------------
def generate_lineups(
    data,
    num_lineups,
    budget,
    min_budget,
    locked,
    stacks,
    max_team,
    min_unique,
    randomness,
):
    lineups = []
    previous_lineup_indices = []
    required_positions = ["PG", "SG", "SF", "PF", "C"]
    present_positions = [
        p for p in required_positions if p in data["Position"].unique()
    ]
    pos_order = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}

    for n in range(num_lineups):
        prob = pulp.LpProblem(f"FanTeam_EuroLeague_Lineup_{n + 1}", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Player", data.index, cat=pulp.LpBinary)

        if randomness > 0 and n > 0:
            jitter = np.random.normal(1.0, randomness, size=len(data))
            current_scores = data["Opt_Score"] * jitter
        else:
            current_scores = data["Opt_Score"]

        # Maximize Opt_Score with FanTeam Tie-Breaker (-0.0001 * Salary favors lower salary on ties)
        prob += (
            pulp.lpSum(
                [
                    (current_scores[i] - 0.0001 * data.loc[i, "Salary"])
                    * player_vars[i]
                    for i in data.index
                ]
            ),
            "Total_Optimization_Score",
        )

        # 1. Roster Size == 7
        prob += (
            pulp.lpSum([player_vars[i] for i in data.index]) == 7,
            "Total_7_Players",
        )

        # 2. Budget Constraints
        prob += (
            pulp.lpSum([data.loc[i, "Salary"] * player_vars[i] for i in data.index])
            <= budget,
            "Max_Budget",
        )
        prob += (
            pulp.lpSum([data.loc[i, "Salary"] * player_vars[i] for i in data.index])
            >= min_budget,
            "Min_Budget",
        )

        # 3. Positional Minimums (At least 1 PG, 1 SG, 1 SF, 1 PF, 1 C)
        for pos in present_positions:
            prob += (
                pulp.lpSum(
                    [
                        player_vars[i]
                        for i in data.index
                        if data.loc[i, "Position"] == pos
                    ]
                )
                >= 1,
                f"Min_1_{pos}",
            )

        # 4. Max Players Per Team
        for team in data["TEAM"].unique():
            prob += (
                pulp.lpSum(
                    [player_vars[i] for i in data.index if data.loc[i, "TEAM"] == team]
                )
                <= max_team,
                f"Max_Team_{team}",
            )

        # 5. Locked Players
        for locked_name in locked:
            locked_idx = data[data["Name"] == locked_name].index.tolist()
            for idx in locked_idx:
                prob += player_vars[idx] == 1, f"Lock_{idx}"

        # 6. Team Stacking Rules
        for team, min_count in stacks.items():
            prob += (
                pulp.lpSum(
                    [player_vars[i] for i in data.index if data.loc[i, "TEAM"] == team]
                )
                >= min_count,
                f"Stack_{team}",
            )

        # 7. Multi-Lineup Uniqueness Constraint
        for idx_list_num, prev_indices in enumerate(previous_lineup_indices):
            prob += (
                pulp.lpSum([player_vars[i] for i in prev_indices]) <= (7 - min_unique),
                f"Unique_{idx_list_num}",
            )

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            break

        selected_indices = [i for i in data.index if player_vars[i].varValue == 1.0]
        previous_lineup_indices.append(selected_indices)

        lineup_df = data.loc[selected_indices].copy()
        lineup_df["Pos_Rank"] = lineup_df["Position"].map(pos_order).fillna(99)
        lineup_df = lineup_df.sort_values(
            by=["Pos_Rank", "Salary"], ascending=[True, False]
        ).drop(columns=["Pos_Rank"])
        lineups.append(lineup_df)

    return lineups


# ---------------------------------------------------------
# GENERATE & RENDER LINEUPS
# ---------------------------------------------------------
st.write("")
if st.button("🚀 BUILD OPTIMAL EUROLEAGUE LINEUPS", type="primary", width="stretch"):
    missing_pos = [
        p
        for p in ["PG", "SG", "SF", "PF", "C"]
        if p not in edited_pool["Position"].unique()
    ]
    if missing_pos:
        st.error(
            f"Cannot build a valid 5+2 lineup! Filtered pool is missing position(s): {', '.join(missing_pos)}"
        )
    elif len(edited_pool) < 7:
        st.error("Not enough players in the active pool to build a 7-player squad.")
    else:
        with st.spinner("Running Linear Programming Solver..."):
            results = generate_lineups(
                data=edited_pool,
                num_lineups=num_lineups,
                budget=budget,
                min_budget=min_budget,
                locked=locked_players,
                stacks=stack_rules,
                max_team=max_per_team,
                min_unique=min_unique,
                randomness=randomness,
            )

        if not results:
            st.warning(
                "No feasible lineups found! Try relaxing your Team Stacks, lowering Min Salary, or unlocking a player."
            )
        else:
            export_rows = []
            for idx, l_df in enumerate(results):
                temp = l_df.copy()
                temp.insert(0, "Lineup_#", idx + 1)
                export_rows.append(temp)
            full_export_df = pd.concat(export_rows, ignore_index=True)

            dl_col1, dl_col2 = st.columns([3, 1])
            with dl_col1:
                st.success(
                    f"✅ Generated {len(results)} Optimal EuroLeague Lineup(s) ({contest_mode})"
                )
            with dl_col2:
                st.download_button(
                    label="📥 Export Lineups to CSV",
                    data=full_export_df.to_csv(index=False).encode("utf-8"),
                    file_name="fanteam_euroleague_lineups.csv",
                    mime="text/csv",
                    width="stretch",
                )

            tab1, tab2 = st.tabs(
                ["🏀 Visual Court & Lineup Breakdown", "📊 Exposure & Stack Analytics"]
            )

            with tab1:
                for i, l_df in enumerate(results):
                    tot_sal = l_df["Salary"].sum()
                    rem_sal = budget - tot_sal
                    tot_proj = l_df["Proj"].sum()
                    tot_floor = l_df["Floor"].sum()
                    tot_ceil = l_df["Ceiling"].sum()
                    tot_opt = l_df["Opt_Score"].sum()

                    team_counts = l_df["TEAM"].value_counts()
                    stack_str = ", ".join(
                        [f"{k} ({v})" for k, v in team_counts.items() if v >= 2]
                    )

                    st.markdown(
                        f"""
                        <div class="lineup-banner">
                            <span style="font-size:1.2rem; font-weight:800; color:#FFFFFF;">
                                🏆 SQUAD #{i + 1}
                            </span>
                            <span style="float:right; color:#FF5A00; font-weight:700;">
                                Stacks: {stack_str if stack_str else "Balanced"}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Salary Used", f"{tot_sal:.1f}M")
                    m2.metric("Tie-Breaker Rem.", f"{rem_sal:.1f}M")
                    m3.metric("Projected Pts", f"{tot_proj:.1f}")
                    m4.metric("Floor / Ceiling", f"{tot_floor:.1f} / {tot_ceil:.1f}")
                    m5.metric("Mode Score", f"{tot_opt:.1f}")

                    # Render 7 Visual Player Court Cards
                    card_cols = st.columns(7)
                    for col_idx, (_, player_row) in enumerate(l_df.iterrows()):
                        with card_cols[col_idx]:
                            st.markdown(
                                f"""
                                <div class="player-card">
                                    <div class="pos-pill">{player_row["Position"]}</div>
                                    <div class="p-name" title="{player_row["Name"]}">{player_row["Name"]}</div>
                                    <div class="p-matchup">{player_row["TEAM"]} vs {player_row["Opp"]}</div>
                                    <div class="p-stats">
                                        <div class="p-stat-box">
                                            <div class="p-stat-label">SAL</div>
                                            <div class="p-stat-val">{player_row["Salary"]:.1f}M</div>
                                        </div>
                                        <div class="p-stat-box">
                                            <div class="p-stat-label">PROJ</div>
                                            <div class="p-stat-proj">{player_row["Proj"]:.1f}</div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with st.expander(f"View Full Stats Table for Squad #{i + 1}"):
                        display_cols = [
                            "Position",
                            "Name",
                            "TEAM",
                            "Opp",
                            "Salary",
                            "Proj",
                            "Floor",
                            "Ceiling",
                            "Value",
                            "Min",
                            "Team_Pts",
                            "Pace",
                            "Win_Prob",
                            "Opt_Score",
                        ]
                        display_cols = [c for c in display_cols if c in l_df.columns]
                        st.dataframe(
                            l_df[display_cols],
                            width="stretch",
                            hide_index=True,
                        )

            with tab2:
                exp_col1, exp_col2 = st.columns([2, 1])
                with exp_col1:
                    st.subheader("Player Exposure Across Generated Squads")
                    exp_df = (
                        full_export_df.groupby(
                            ["Name", "Position", "TEAM", "Salary", "Proj"]
                        )
                        .size()
                        .reset_index(name="Squads")
                    )
                    exp_df["Exposure_%"] = (
                        (exp_df["Squads"] / len(results)) * 100
                    ).round(1)
                    exp_df = exp_df.sort_values(
                        by=["Exposure_%", "Proj"], ascending=[False, False]
                    )
                    st.dataframe(
                        exp_df,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Salary": st.column_config.NumberColumn(
                                "Salary", format="%.1fM"
                            ),
                            "Proj": st.column_config.NumberColumn(
                                "Proj", format="%.1f"
                            ),
                            "Exposure_%": st.column_config.ProgressColumn(
                                "Exposure %",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                    )
                with exp_col2:
                    st.subheader("Team Exposure")
                    team_exp = (
                        full_export_df.groupby("TEAM")
                        .size()
                        .reset_index(name="Total_Roster_Spots")
                    )
                    team_exp["Avg_Players_Per_Lineup"] = (
                        team_exp["Total_Roster_Spots"] / len(results)
                    ).round(2)
                    team_exp = team_exp.sort_values(
                        by="Total_Roster_Spots", ascending=False
                    )
                    st.dataframe(
                        team_exp,
                        width="stretch",
                        hide_index=True,
                    )
