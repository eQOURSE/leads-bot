"""Phase 10 — Reusable chart/data builders for the dashboard.

Uses Streamlit-native charts (no plotly dependency). Functions here return
DataFrames shaped for st.bar_chart / st.line_chart, plus a text funnel helper.
"""

from __future__ import annotations

import pandas as pd


def funnel_dataframe(funnel: dict[str, int]) -> pd.DataFrame:
    """Shape a funnel dict into a DataFrame for st.bar_chart.

    Preserves stage order via a categorical index.
    """
    stages = list(funnel.keys())
    df = pd.DataFrame(
        {"stage": stages, "count": [funnel[s] for s in stages]}
    ).set_index("stage")
    return df


def funnel_text(funnel: dict[str, int]) -> str:
    """A compact text representation of the funnel for quick reading."""
    labels = {
        "candidates": "Candidates",
        "qualified": "Qualified",
        "ready_to_send": "Ready to send",
    }
    parts = []
    for stage, count in funnel.items():
        parts.append(f"{labels.get(stage, stage)}: {count}")
    return "  →  ".join(parts)


def runs_per_day(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Count of runs grouped by calendar day (from started_at)."""
    if runs_df.empty or "started_at" not in runs_df.columns:
        return pd.DataFrame(columns=["count"])
    s = pd.to_datetime(runs_df["started_at"], errors="coerce").dt.date
    counts = s.value_counts().sort_index()
    return pd.DataFrame({"count": counts.values}, index=pd.Index(counts.index, name="day"))


def qualified_per_day(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Sum of qualified_count grouped by calendar day."""
    if runs_df.empty or "started_at" not in runs_df.columns:
        return pd.DataFrame(columns=["qualified"])
    df = runs_df.copy()
    df["day"] = pd.to_datetime(df["started_at"], errors="coerce").dt.date
    df["qualified_count"] = pd.to_numeric(df.get("qualified_count"), errors="coerce").fillna(0)
    agg = df.groupby("day")["qualified_count"].sum()
    return pd.DataFrame({"qualified": agg.values}, index=pd.Index(agg.index, name="day"))
