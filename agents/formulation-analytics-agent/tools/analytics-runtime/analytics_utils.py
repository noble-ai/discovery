"""Small helpers for formulation analytics scripts in the coding environment."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


def _parse_list_value(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return value


def _column_contains_lists(series: pd.Series) -> bool:
    for value in series.dropna().head(50):
        parsed = _parse_list_value(value)
        if isinstance(parsed, list) and parsed is not value:
            return True
        if isinstance(value, (list, tuple)):
            return True
    return False


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stringified list cells parsed into Python lists."""
    out = df.copy()
    for column in out.columns:
        if _column_contains_lists(out[column]):
            out[column] = out[column].map(_parse_list_value)
    return out


def write_plotly_html(fig, path: str | Path) -> str:
    """Write a self-contained Plotly HTML file and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(destination), include_plotlyjs=True)
    return str(destination)
