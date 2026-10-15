"""Unit tests for analytics coding-environment helpers. No container required."""

from types import SimpleNamespace

import pandas as pd

from analytics_utils import parse_list_columns, write_plotly_html


def test_parse_list_columns_parses_stringified_pairs():
    frame = pd.DataFrame(
        {
            "IDs": ['[240, 777]', "[1]"],
            "Proportions": ["[0.2, 0.8]", "[1.0]"],
            "logV": [1.2, 0.4],
        }
    )

    parsed = parse_list_columns(frame)

    assert parsed["IDs"].tolist() == [[240, 777], [1]]
    assert parsed["Proportions"].tolist() == [[0.2, 0.8], [1.0]]
    assert frame["IDs"].tolist() == ['[240, 777]', "[1]"]


def test_parse_list_columns_leaves_scalar_columns():
    frame = pd.DataFrame({"T": [25, 40], "note": ["a", "b"]})

    parsed = parse_list_columns(frame)

    assert parsed["T"].tolist() == [25, 40]
    assert parsed["note"].tolist() == ["a", "b"]


def test_write_plotly_html_inlines_plotly_js(tmp_path):
    calls = {}

    def write_html(path, include_plotlyjs):
        calls["path"] = path
        calls["include_plotlyjs"] = include_plotlyjs

    fig = SimpleNamespace(write_html=write_html)
    destination = tmp_path / "charts" / "composition.html"

    result = write_plotly_html(fig, destination)

    assert result == str(destination)
    assert calls == {"path": str(destination), "include_plotlyjs": True}
    assert destination.parent.is_dir()
