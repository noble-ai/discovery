#!/usr/bin/env python3
"""Deterministic create_chart action for the formulation analytics runtime.

Writes a self-contained Plotly HTML file and chart_result.json. Scripts in the
coding environment prepare tables; they do not build figures.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from analytics_utils import parse_list_columns, write_plotly_html

_SIZE_LIMITS = {
    "bar": 50,
    "box": 30,
    "pie": 20,
    "hbar": 30,
}

_CHART_TYPES = (
    "scatter",
    "bar",
    "histogram",
    "box",
    "pie",
    "hbar",
    "heatmap",
    "density_heatmap",
)


@dataclass
class ChartArgs:
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    z: Optional[str] = None
    color: Optional[str] = None
    title: Optional[str] = None
    orientation: Optional[str] = None
    names: Optional[str] = None
    values: Optional[str] = None
    barmode: Optional[str] = None
    marginal: str = "box"
    parity_line: bool = False
    plotly_kwargs: Optional[str] = None
    layout_kwargs: Optional[str] = None
    chart_name: Optional[str] = None
    data_path: str = ""


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _blank_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _slug(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    return slug or "chart"


def _chart_stem(chart_name: Optional[str], title: Optional[str], chart_type: str) -> str:
    if chart_name:
        stem = Path(chart_name).name
    elif title:
        stem = _slug(title)
    else:
        stem = chart_type
    if stem.lower().endswith(".html"):
        stem = stem[:-5]
    stem = stem.strip() or chart_type
    return stem


def _check_size_limit(df: pd.DataFrame, chart_type: str, args: ChartArgs) -> Optional[str]:
    """Return the size-limit error text when the chart would be too crowded."""
    limit = _SIZE_LIMITS.get(chart_type)
    if limit is None:
        return None

    if chart_type == "bar" and args.x:
        count = df[args.x].nunique()
        element_name = "bars"
    elif chart_type == "hbar" and args.y:
        count = df[args.y].nunique()
        element_name = "horizontal bars"
    elif chart_type == "pie" and args.names:
        count = df[args.names].nunique()
        element_name = "slices"
    elif chart_type == "box" and args.x:
        count = df[args.x].nunique()
        element_name = "boxes"
    elif chart_type == "box":
        return None
    else:
        count = len(df)
        element_name = "elements"

    if count > limit:
        return (
            f"Chart has too many {element_name} ({count}, limit is {limit}). "
            f"Explain to user and ask how they want to reduce: "
            f"filter by criteria, sample rows, or aggregate."
        )
    return None


def _is_numeric(series: pd.Series) -> bool:
    return series.dtype in [np.float64, np.int64, float, int]


def _extreme_value_note(series: pd.Series, col_name: str) -> Optional[str]:
    """Note about values beyond 1.5x IQR (standard box-plot whisker criterion)."""
    clean = series.dropna()
    if not _is_numeric(clean) or len(clean) < 4:
        return None
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    n_extreme = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
    if n_extreme > 0:
        pct = 100 * n_extreme / len(clean)
        return (
            f"{col_name} has {n_extreme} extreme values "
            f"({pct:.1f}% of data, beyond 1.5xIQR)."
        )
    return None


def _compute_chart_context(df: pd.DataFrame, chart_type: str, args: ChartArgs) -> str:
    """Compact stats string so the agent can describe the chart without the figure JSON."""
    x = args.x
    y = args.y
    color = args.color
    title = args.title
    n_rows = len(df)

    parts = [f"Created {chart_type} chart"]
    if title:
        parts.append(f"titled '{title}'")
    parts.append(f"with {n_rows} data points.")

    plotted_cols = [
        c for c in [x, y, args.z, color, args.names, args.values] if c and c in df.columns
    ]
    missing_notes = []
    for col in plotted_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            missing_notes.append(f"{col}: {n_missing}")
    if missing_notes:
        parts.append(f"Missing values — {', '.join(missing_notes)}.")

    try:
        if chart_type == "scatter" and x and y:
            if _is_numeric(df[x]) and _is_numeric(df[y]):
                corr = df[x].corr(df[y])
                if not np.isnan(corr):
                    direction = (
                        "positive" if corr > 0.1 else "negative" if corr < -0.1 else "no clear"
                    )
                    strength = (
                        "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                    )
                    parts.append(f"X-axis: {x} (range: {df[x].min():.2f} to {df[x].max():.2f}).")
                    parts.append(f"Y-axis: {y} (range: {df[y].min():.2f} to {df[y].max():.2f}).")
                    parts.append(f"Shows {strength} {direction} correlation (r={corr:.2f}).")
            for col in [x, y]:
                note = _extreme_value_note(df[col], col)
                if note:
                    parts.append(note)
            if color:
                n_groups = df[color].nunique()
                parts.append(f"Colored by {color} ({n_groups} groups).")

        elif chart_type == "histogram" and x:
            if _is_numeric(df[x]):
                mean_val = df[x].mean()
                std_val = df[x].std()
                skew = df[x].skew()
                skew_desc = (
                    "right-skewed"
                    if skew > 0.5
                    else "left-skewed"
                    if skew < -0.5
                    else "approximately symmetric"
                )
                parts.append(
                    f"Distribution of {x}: mean={mean_val:.2f}, std={std_val:.2f}, {skew_desc}."
                )
                note = _extreme_value_note(df[x], x)
                if note:
                    parts.append(note)

        elif chart_type == "bar" and x and y:
            parts.append(f"X-axis: {x}, Y-axis: {y}.")
            if _is_numeric(df[y]):
                top_n = min(3, len(df))
                top_3 = df.nlargest(top_n, y)[[x, y]]
                top_items = ", ".join(f"{row[x]}={row[y]:.2f}" for _, row in top_3.iterrows())
                parts.append(f"Top values: {top_items}.")
                bottom_3 = df.nsmallest(top_n, y)[[x, y]]
                bottom_items = ", ".join(
                    f"{row[x]}={row[y]:.2f}" for _, row in bottom_3.iterrows()
                )
                parts.append(f"Bottom values: {bottom_items}.")

        elif chart_type == "box" and y:
            parts.append(f"Box plot of {y}.")
            if _is_numeric(df[y]):
                median = df[y].median()
                q1, q3 = df[y].quantile([0.25, 0.75])
                parts.append(f"Median={median:.2f}, IQR=[{q1:.2f}, {q3:.2f}].")
                note = _extreme_value_note(df[y], y)
                if note:
                    parts.append(note)
            if x:
                n_categories = df[x].nunique()
                parts.append(f"Grouped by {x} ({n_categories} categories).")

        elif chart_type == "pie":
            names_col = args.names
            values_col = args.values
            if names_col and values_col:
                top_3 = df.nlargest(min(3, len(df)), values_col)[[names_col, values_col]]
                total = df[values_col].sum()
                if total > 0:
                    top_items = ", ".join(
                        f"{row[names_col]}={100 * row[values_col] / total:.1f}%"
                        for _, row in top_3.iterrows()
                    )
                    parts.append(f"Top slices: {top_items}.")
                    top_pct = 100 * top_3[values_col].sum() / total
                    remaining_pct = 100 - top_pct
                    n_remaining = df[names_col].nunique() - len(top_3)
                    if n_remaining > 0:
                        parts.append(
                            f"Remaining {n_remaining} slices account for {remaining_pct:.1f}%."
                        )

        elif chart_type == "hbar" and x and y:
            parts.append(f"Horizontal bar: {x} by {y}.")
            n_items = df[y].nunique()
            parts.append(f"Showing {n_items} items.")
            if color:
                n_colors = df[color].nunique()
                parts.append(f"Stacked by {color} ({n_colors} components).")

        elif chart_type == "heatmap" and x and y and args.z:
            parts.append(f"Grid axes: {x} by {y}, values: {args.z}.")
            if _is_numeric(df[args.z]):
                parts.append(f"{args.z} range: {df[args.z].min():.2f} to {df[args.z].max():.2f}.")

        elif chart_type == "density_heatmap" and x and y:
            parts.append(f"Density of {x} by {y}.")
            for axis_name, col in [("X", x), ("Y", y)]:
                if _is_numeric(df[col]):
                    parts.append(
                        f"{axis_name}-axis range: {df[col].min():.2f} to {df[col].max():.2f}."
                    )

    except Exception:
        pass

    return " ".join(parts)


def _parse_json_object(raw: Optional[str], label: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid JSON in {label}: expected an object")
    return parsed


def _build_figure(df: pd.DataFrame, args: ChartArgs, plotly_kwargs: dict[str, Any]):
    import plotly.express as px

    if args.chart_type == "scatter":
        if not args.x or not args.y:
            raise ValueError("Scatter chart requires 'x' and 'y' parameters")
        return px.scatter(df, x=args.x, y=args.y, color=args.color, title=args.title, **plotly_kwargs)

    if args.chart_type == "bar":
        if not args.x or not args.y:
            raise ValueError("Bar chart requires 'x' and 'y' parameters")
        return px.bar(
            df,
            x=args.x,
            y=args.y,
            color=args.color,
            title=args.title,
            orientation=args.orientation or "v",
            barmode=args.barmode,
            **plotly_kwargs,
        )

    if args.chart_type == "histogram":
        if not args.x:
            raise ValueError("Histogram requires 'x' parameter")
        return px.histogram(
            df,
            x=args.x,
            color=args.color,
            title=args.title,
            marginal=None if args.marginal == "none" else args.marginal,
            barmode=args.barmode,
            **plotly_kwargs,
        )

    if args.chart_type == "box":
        if not args.y:
            raise ValueError("Box chart requires 'y' parameter")
        return px.box(df, x=args.x, y=args.y, color=args.color, title=args.title, **plotly_kwargs)

    if args.chart_type == "pie":
        if not args.names or not args.values:
            raise ValueError("Pie chart requires 'names' and 'values' parameters")
        fig = px.pie(df, names=args.names, values=args.values, title=args.title, **plotly_kwargs)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    if args.chart_type == "hbar":
        if not args.x or not args.y:
            raise ValueError("Horizontal bar chart requires 'x' and 'y' parameters")
        return px.bar(
            df,
            x=args.x,
            y=args.y,
            color=args.color,
            title=args.title,
            orientation="h",
            barmode=args.barmode or "stack",
            **plotly_kwargs,
        )

    if args.chart_type == "heatmap":
        if not args.x or not args.y or not args.z:
            raise ValueError("Heatmap requires 'x', 'y', and 'z' parameters")
        matrix = df.pivot(index=args.y, columns=args.x, values=args.z)
        return px.imshow(matrix, title=args.title, **plotly_kwargs)

    if args.chart_type == "density_heatmap":
        if not args.x or not args.y:
            raise ValueError("Density heatmap requires 'x' and 'y' parameters")
        return px.density_heatmap(df, x=args.x, y=args.y, title=args.title, **plotly_kwargs)

    raise ValueError(f"Unknown chart type: {args.chart_type}")


def _resolve_data_path(data_path: str, input_dir: str) -> Path:
    path = Path(data_path)
    if not path.is_absolute():
        path = Path(input_dir) / path
    return path


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return parse_list_columns(pd.read_csv(path))
    raise ValueError(f"Unsupported data file type '{path.suffix}'. Use .csv or .parquet.")


def _write_result(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chart_result.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _fail(output_dir: Path, message: str) -> None:
    _write_result(output_dir, {"status": "failed", "message": message})
    print(message)
    raise SystemExit(1)


def _args_payload(args: ChartArgs, plotly_kwargs: dict[str, Any], layout_kwargs: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data_path": args.data_path,
        "chart_type": args.chart_type,
        "marginal": args.marginal,
        "parity_line": args.parity_line,
    }
    for key in ("x", "y", "z", "color", "title", "names", "values", "orientation", "barmode", "chart_name"):
        value = getattr(args, key)
        if value is not None:
            payload[key] = value
    if plotly_kwargs:
        payload["plotly_kwargs"] = plotly_kwargs
    if layout_kwargs:
        payload["layout_kwargs"] = layout_kwargs
    return payload


def _error_message(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return f"Column '{exc.args[0]}' was not found in the data."
    if isinstance(exc, ValueError):
        return str(exc)
    return f"Error creating chart: {exc}"


def _run(ns: argparse.Namespace) -> None:
    output_dir = Path(ns.output)
    args = ChartArgs(
        chart_type=ns.chart_type,
        x=_blank_to_none(ns.x),
        y=_blank_to_none(ns.y),
        z=_blank_to_none(ns.z),
        color=_blank_to_none(ns.color),
        title=_blank_to_none(ns.title),
        orientation=_blank_to_none(ns.orientation),
        names=_blank_to_none(ns.names),
        values=_blank_to_none(ns.values),
        barmode=_blank_to_none(ns.barmode),
        marginal=_blank_to_none(ns.marginal) or "box",
        parity_line=_parse_bool(ns.parity_line, False),
        plotly_kwargs=_blank_to_none(ns.plotly_kwargs),
        layout_kwargs=_blank_to_none(ns.layout_kwargs),
        chart_name=_blank_to_none(ns.chart_name),
        data_path=ns.data_path,
    )

    try:
        frame = _load_frame(_resolve_data_path(ns.data_path, ns.input))
        size_message = _check_size_limit(frame, args.chart_type, args)
    except Exception as exc:
        _fail(output_dir, _error_message(exc))
    if size_message:
        _fail(output_dir, size_message)

    try:
        plotly_kwargs = _parse_json_object(args.plotly_kwargs, "plotly_kwargs")
        layout_kwargs = _parse_json_object(args.layout_kwargs, "layout_kwargs")
        fig = _build_figure(frame, args, plotly_kwargs)
        fig.update_layout(template="plotly_white")
        fig.update_layout(**layout_kwargs)
        # Added after layout_kwargs: update_layout(shapes=...) replaces all shapes.
        if args.parity_line and args.chart_type == "scatter" and args.x and args.y:
            lo = float(min(frame[args.x].min(), frame[args.y].min()))
            hi = float(max(frame[args.x].max(), frame[args.y].max()))
            fig.add_shape(
                type="line",
                x0=lo,
                y0=lo,
                x1=hi,
                y1=hi,
                line={"dash": "dash", "color": "gray"},
                row="all",
                col="all",
            )
        chart_context = _compute_chart_context(frame, args.chart_type, args)
    except Exception as exc:
        _fail(output_dir, _error_message(exc))

    stem = _chart_stem(args.chart_name, args.title, args.chart_type)
    chart_path = write_plotly_html(fig, output_dir / "charts" / f"{stem}.html")
    title = args.title or f"{args.chart_type} chart"
    _write_result(
        output_dir,
        {
            "status": "success",
            "chart_path": chart_path,
            "chart_type": args.chart_type,
            "title": title,
            "chart_context": chart_context,
            "args": _args_payload(args, plotly_kwargs, layout_kwargs),
        },
    )
    print(f"chart_path: {chart_path}")
    print(f"chart_context: {chart_context}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Plotly HTML chart from CSV or parquet.")
    parser.add_argument("--input", required=True, help="Directory used to resolve a relative data path.")
    parser.add_argument("--output", required=True, help="Directory for charts/ and chart_result.json.")
    parser.add_argument("--data-path", required=True, help="CSV or parquet path, absolute or relative to --input.")
    parser.add_argument("--chart-type", required=True, choices=_CHART_TYPES)
    parser.add_argument("--x")
    parser.add_argument("--y")
    parser.add_argument("--z")
    parser.add_argument("--color")
    parser.add_argument("--title")
    parser.add_argument("--names")
    parser.add_argument("--values")
    parser.add_argument("--orientation", choices=("v", "h"))
    parser.add_argument("--barmode", choices=("group", "stack", "overlay", "relative"))
    parser.add_argument("--marginal", default="box", choices=("box", "violin", "rug", "histogram", "none"))
    parser.add_argument("--parity-line", default=None)
    parser.add_argument("--plotly-kwargs")
    parser.add_argument("--layout-kwargs")
    parser.add_argument("--chart-name")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    _run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
