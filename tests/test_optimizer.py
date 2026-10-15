# cspell:disable
import pandas as pd

from app import (
    REQUIRED_POSITIONS,
    SLOT_LABELS,
    compute_metrics_after_edit,
    estimate_ownership,
    generate_lineups,
    get_sample_data,
    normalize_dataframe,
    repair_slate_anomalies,
)


def _prepared_df() -> pd.DataFrame:
    df = normalize_dataframe(get_sample_data())
    df = repair_slate_anomalies(df)
    return compute_metrics_after_edit(
        df=df,
        contest_mode="GPP Tournament (Ceiling / Upside)",
        blend_weight=0.45,
        env_boost=True,
        blowout_threshold=10.0,
        blowout_penalty_pct=5.0,
        own_penalty_lambda=0.05,
        use_estimated_own=True,
        own_sharpness=1.6,
    )


def test_ownership_sums_to_700_percent():
    df = normalize_dataframe(get_sample_data())
    own = estimate_ownership(df, sharpness=1.6)
    assert len(own) == len(df)
    assert 680.0 <= own.sum() <= 720.0


def test_roster_size_budget_and_slots():
    df = _prepared_df()
    lineups = generate_lineups(
        data=df,
        num_lineups=3,
        budget=75.0,
        min_budget=70.0,
        locked=[],
        stacks={},
        bring_back_enabled=False,
        auto_bring_back_min=0,
        max_team=4,
        max_per_game=5,
        min_unique=2,
        randomness=0.0,
        team_shock_share=0.0,
        max_lineup_own=700.0,
        max_chalk_count=7,
        chalk_threshold=50.0,
        force_pairs=[],
        ban_pairs=[],
    )
    assert len(lineups) == 3
    for l_df in lineups:
        assert len(l_df) == 7
        assert 70.0 <= l_df["Salary"].sum() <= 75.0 + 1e-5
        assert l_df["Slot"].tolist() == SLOT_LABELS
        for pos in REQUIRED_POSITIONS:
            assert (l_df["Position"] == pos).sum() >= 1


def test_stack_bring_back_and_pair_rules():
    df = _prepared_df()
    lineups = generate_lineups(
        data=df,
        num_lineups=2,
        budget=75.0,
        min_budget=65.0,
        locked=[],
        stacks={"PBB": 3},
        bring_back_enabled=True,
        auto_bring_back_min=3,
        max_team=4,
        max_per_game=6,
        min_unique=1,
        randomness=0.0,
        team_shock_share=0.0,
        max_lineup_own=700.0,
        max_chalk_count=7,
        chalk_threshold=50.0,
        force_pairs=[("TJ Shorts", "Nadir Hifi")],
        ban_pairs=[("Sasha Vezenkov", "Evan Fournier")],
    )
    assert len(lineups) >= 1
    for l_df in lineups:
        names = set(l_df["Name"])
        assert (l_df["TEAM"] == "PBB").sum() >= 3
        assert (l_df["TEAM"] == "VIR").sum() >= 1
        assert "TJ Shorts" in names and "Nadir Hifi" in names
        assert not ("Sasha Vezenkov" in names and "Evan Fournier" in names)
