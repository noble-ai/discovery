"""Tests for the create_chart action. No container required."""

import json
from pathlib import Path

import pandas as pd
import pytest

from create_chart import main


def _write_csv(directory: Path, frame: pd.DataFrame, name: str = "data.csv") -> Path:
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def _result(output: Path) -> dict:
    return json.loads((output / "chart_result.json").read_text(encoding="utf-8"))


def _run(argv: list[str]) -> None:
    main(argv)


def test_each_chart_type_writes_html_and_context(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    scatter = pd.DataFrame(
        {
            "T": [1, 2, 3, 4, 5, 6],
            "logV": [1.0, 2.1, 2.9, 4.2, 4.8, 6.0],
            "group": ["a", "a", "b", "b", "a", "b"],
        }
    )
    categories = pd.DataFrame(
        {
            "cat": ["a", "b", "c"],
            "amount": [3.0, 1.0, 2.0],
            "item": ["r1", "r2", "r3"],
            "component": ["x", "y", "x"],
        }
    )
    grid = pd.DataFrame(
        {
            "x": ["a", "a", "b", "b"],
            "y": ["p", "q", "p", "q"],
            "z": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cases = {
        "scatter": (scatter, ["--x", "T", "--y", "logV", "--color", "group"]),
        "bar": (categories, ["--x", "cat", "--y", "amount"]),
        "histogram": (scatter, ["--x", "T", "--marginal", "none"]),
        "box": (scatter, ["--y", "logV", "--x", "group"]),
        "pie": (categories, ["--names", "cat", "--values", "amount"]),
        "hbar": (categories, ["--x", "amount", "--y", "item", "--color", "component"]),
        "heatmap": (grid, ["--x", "x", "--y", "y", "--z", "z"]),
        "density_heatmap": (scatter, ["--x", "T", "--y", "logV"]),
    }

    for chart_type, (frame, extra) in cases.items():
        chart_input = source / chart_type
        chart_input.mkdir()
        chart_output = output / chart_type
        _write_csv(chart_input, frame)
        _run(
            [
                "--input",
                str(chart_input),
                "--output",
                str(chart_output),
                "--data-path",
                "data.csv",
                "--chart-type",
                chart_type,
                "--title",
                f"{chart_type} example",
                *extra,
            ]
        )
        result = _result(chart_output)
        html_path = Path(result["chart_path"])
        assert result["status"] == "success"
        assert result["chart_type"] == chart_type
        assert result["chart_context"]
        assert html_path.is_file()
        assert html_path.parent == chart_output / "charts"
        assert "plotly" in html_path.read_text(encoding="utf-8").lower()


def test_scatter_context_reports_correlation_and_title_slug(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    _write_csv(
        source,
        pd.DataFrame({"T": [1, 2, 3, 4, 5], "logV": [1.0, 2.0, 3.0, 4.0, 5.0]}),
    )

    _run(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--data-path",
            "data.csv",
            "--chart-type",
            "scatter",
            "--x",
            "T",
            "--y",
            "logV",
            "--title",
            "Temperature vs Viscosity",
        ]
    )

    result = _result(output)
    assert "r=" in result["chart_context"]
    assert Path(result["chart_path"]).name == "temperature_vs_viscosity.html"


def test_parity_line_adds_shape(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    _write_csv(
        source,
        pd.DataFrame({"actual": [1.0, 2.0, 3.0, 4.0], "predicted": [1.2, 1.8, 3.4, 3.6]}),
    )

    _run(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--data-path",
            "data.csv",
            "--chart-type",
            "scatter",
            "--x",
            "actual",
            "--y",
            "predicted",
            "--parity-line",
            "true",
            "--chart-name",
            "parity",
        ]
    )

    html = Path(_result(output)["chart_path"]).read_text(encoding="utf-8")
    assert '"dash":"dash"' in html or '"dash": "dash"' in html
    assert "gray" in html


def test_bar_over_size_limit_exits(tmp_path, capsys):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    _write_csv(
        source,
        pd.DataFrame({"cat": [f"c{i}" for i in range(51)], "val": [1.0] * 51}),
    )

    with pytest.raises(SystemExit) as exc:
        _run(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--data-path",
                "data.csv",
                "--chart-type",
                "bar",
                "--x",
                "cat",
                "--y",
                "val",
            ]
        )

    assert exc.value.code == 1
    result = _result(output)
    assert result["status"] == "failed"
    assert "too many bars (51, limit is 50)" in result["message"]
    assert "Explain to user and ask how they want to reduce" in result["message"]
    assert result["message"] in capsys.readouterr().out
    assert not (output / "charts").exists()


def test_invalid_plotly_kwargs_json_fails(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    _write_csv(source, pd.DataFrame({"T": [1, 2, 3], "logV": [1.0, 2.0, 3.0]}))

    with pytest.raises(SystemExit) as exc:
        _run(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--data-path",
                "data.csv",
                "--chart-type",
                "scatter",
                "--x",
                "T",
                "--y",
                "logV",
                "--plotly-kwargs",
                "{bad",
            ]
        )

    assert exc.value.code == 1
    message = _result(output)["message"]
    assert message.startswith("Invalid JSON in plotly_kwargs:")


def test_pie_without_names_fails(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    _write_csv(source, pd.DataFrame({"cat": ["a", "b"], "amount": [1.0, 2.0]}))

    with pytest.raises(SystemExit) as exc:
        _run(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--data-path",
                "data.csv",
                "--chart-type",
                "pie",
                "--values",
                "amount",
            ]
        )

    assert exc.value.code == 1
    assert "Pie chart requires 'names' and 'values' parameters" in _result(output)["message"]


def test_parquet_input_and_relative_data_path(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    frame = pd.DataFrame({"T": [10, 20, 30, 40], "logV": [1.0, 1.5, 2.0, 2.5]})
    frame.to_parquet(source / "prepared.parquet", index=False)

    _run(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--data-path",
            "prepared.parquet",
            "--chart-type",
            "scatter",
            "--x",
            "T",
            "--y",
            "logV",
            "--chart-name",
            "from_parquet",
        ]
    )

    result = _result(output)
    assert result["status"] == "success"
    assert result["args"]["data_path"] == "prepared.parquet"
    assert "r=" in result["chart_context"]
    assert Path(result["chart_path"]).name == "from_parquet.html"
