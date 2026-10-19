# NobleAI Formulation Analytics Agent

Explores chemical formulation and tabular experimental CSVs on Microsoft Discovery. A Python coding environment profiles data and computes statistics. The `create_chart` action writes self-contained Plotly HTML charts. It does not train models.

## Overview

Chemists and materials scientists often need to inspect formulation tables (component IDs and proportions, targets, process covariates) before any modeling run. This agent answers those exploratory questions in chat: schema and missingness, composition views, correlations, simple cleaning, and a short model-readiness narrative.

It is **not** the NobleAI Formulation Modeling Agent. Modeling (Hydra YAML, training, sweeps, predict) stays on that agent. Use this agent for analysis and charts; switch to formulation modeling with the same uploaded CSVs when you are ready to train.

This version is a prompt agent plus one container that exposes a coding environment and a `create_chart` action. Charts are self-contained HTML files: the action writes them, and the user previews or downloads the file. The agent does not use Foundry Code Interpreter, table widgets, reports, memory, suggested queries, or deep-analysis mode.

## Architecture

```
Linked CSV storage asset
  → mount parent at /inputs (optional prior output at /state)
  → optional Python script prepares /output/data/*.parquet
  → create_chart action
  → /output/charts/*.html, /output/chart_result.json, /output/tables/*.csv
  → PreviewResource / ShareResource
  → markdown takeaways in chat
```

- **Model**: workspace chat deployment (`{{CHAT-MODEL}}`).
- **Compute**: container image `formulation-analytics:0.2.0` (pandas, scipy, scikit-learn, statsmodels, pyarrow, plotly). Discovery starts the container and mounts files. There is no nested sandbox and no persistent Python kernel.
- **Charts**: the `create_chart` action saves self-contained HTML (`include_plotlyjs=True`) under `/output/charts` and writes `/output/chart_result.json`. Discovery chat does not render an inline Plotly widget; the user previews or downloads the HTML file. Scripts prepare tables and do not import Plotly.
- **Tables**: small summaries as markdown in the assistant reply; large results as CSV files under `/output/tables`.
- **State**: variables die when the job exits. Reusable and chart tables are Parquet under `/output/data`. A later job remounts that output at `/state` (`/state/data/<name>.parquet`) only when it needs them.
- **Handoff**: if the user asks to train or predict, the agent points them at the formulation modeling agent. It does not author training YAML.

## Prerequisites

- Microsoft Discovery workspace with a deployed project
- Azure AI Foundry chat model deployment (`{{CHAT-MODEL}}`)
- Azure Container Registry for the analytics image
- Formulation or tabular CSVs registered as Discovery storage assets (Studio does not upload local files)

Sample formulation CSVs used by the modeling agent (do not need to be copied here):

- `agents/formulation-modeling-agent/tools/ml-core/example-input-files/basic/formulation_sample.csv`
- `agents/formulation-modeling-agent/tools/ml-core/example-input-files/basic/components_sample.csv`

## Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{{CHAT-MODEL}}` | Azure AI Foundry model deployment name | `gpt-4o` |
| `{{analyticsRuntimeToolId}}` | ARM resource ID of the deployed analytics-runtime tool | set at deploy time |

Build and register `tools/analytics-runtime` before the agent can run scripts or `create_chart`. The catalog deployer builds this image when the agent is deployed from the repo. The image tag is `formulation-analytics:0.2.0` so Discovery does not reuse an older cached image.

## Usage

1. Deploy the agent and the analytics-runtime tool.
2. Link a formulation CSV (and optional component CSV) as a storage asset on the session.
3. Ask an exploratory question. The agent mounts the parent directory at `/inputs`. A chart of raw columns calls `create_chart` directly. A chart that needs a filter, aggregate, or explode uses one coding job, then `create_chart` on the remounted parquet. The agent then previews and shares the HTML (and CSVs when needed).

Example prompts (documentation only; this agent does not show suggestion chips):

| Prompt | Input | Expected output |
|--------|--------|-----------------|
| Profile my formulation dataset | formulation CSV | Column types, missingness, list-column alignment, short markdown summary |
| Show composition for the first 20 formulations | formulation CSV | Coding job writes `/output/data/chart_data.parquet`; `create_chart` writes `/output/charts/composition_hbar.html` |
| How does temperature relate to viscosity? | CSV with T and a viscosity column | `create_chart` scatter HTML plus correlation / range stats from `chart_context` |
| What is missing, and is this ready to model viscosity? | formulation CSV, optional component CSV | Missingness notes and readiness *advice*; no YAML and no training |

After a chart is shared, open the HTML file to interact with the Plotly figure.

When you want a trained property model, use **NobleAI Formulation Modeling Agent** with the same files.

## Tools

### analytics-runtime

Coding environment plus one execute action, `create_chart`.

| Path | Role |
|------|------|
| `/inputs` | Parent directory of uploaded CSVs |
| `/state` | Optional remount of a previous job's `/output` |
| `/output` | Chart HTML, `chart_result.json`, CSV tables, parquet under `/output/data`, and logs |
| `/output/data/<name>.parquet` | Prepared chart table. Remounted path is `/state/data/<name>.parquet` |
| `/output/charts/<name>.html` | Self-contained Plotly chart from `create_chart` |

Helpers in the image: `parse_list_columns`. `write_plotly_html` remains for the action; scripts do not call it.

The generated script is run with:

```text
python3 -u "/{{scriptName}}" > {{output_directory}}/logs.out 2>&1; tail -n 100 {{output_directory}}/logs.out
```

Stdout in the agent reply is the last 100 log lines. Put results the user should keep in files under `/output`.

### create_chart

| Input | Required | Role |
|-------|----------|------|
| `input_directory` | yes | `/inputs` for a raw upload, or `/state` for a remounted coding job |
| `output_directory` | yes | Typically `/output` |
| `data_path` | yes | Absolute or relative CSV or parquet path |
| `chart_type` | yes | `scatter`, `bar`, `histogram`, `box`, `pie`, `hbar`, `heatmap`, `density_heatmap` |
| `x`, `y`, `z`, `color` | per type | Column names. Heatmap tick labels (`x`, `y`) should be short |
| `names`, `values` | pie | Slice labels and sizes |
| `title`, `chart_name` | no | Title, and HTML file stem. Reuse `chart_name` to replace a chart |
| `orientation`, `barmode` | no | Bar layout. `hbar` defaults to `stack` |
| `marginal` | no | Histogram marginal: `box` (default), `violin`, `rug`, `histogram`, or `none` |
| `parity_line` | no | `true` draws a dashed y = x line on a scatter |
| `plotly_kwargs`, `layout_kwargs` | no | JSON objects (double quotes). Do not set width or height unless asked |

Outputs:

- `{output_directory}/charts/<chart_name or slug(title) or chart_type>.html`
- `{output_directory}/chart_result.json` with `status`, `chart_path`, `chart_type`, `title`, `chart_context`, and `args`
- stdout: `chart_path` and `chart_context` on success; the error message on failure (exit code 1)

Size limits: bar 50, hbar 30, pie 20, box 30. Scatter and histogram are unlimited. An over-limit chart fails and asks the user whether to filter, sample, or aggregate.

## Known Limitations

- Charts are HTML files, not inline Plotly widgets.
- Column names, titles, `chart_name`, and JSON arguments cannot contain a single quote (`'`). Rename those columns in a coding job first. JSON uses double quotes.
- Bar, horizontal bar, pie, and box charts fail when they exceed the size limits above. The agent must ask the user how to reduce the data before retrying.
- Each coding job is a new process. Assigning a variable does not make it available on the next call. Persist Parquet and JSON under `/output/data` and remount that output at `/state`.
- One tool call at a time. Do inspection and chart-data preparation in one coding job, then call `create_chart` once per chart.
- Logs returned to the agent from a coding job are truncated to the last 100 lines.
- Human-in-the-loop (`GetUserInput`) is enabled for ambiguous questions and for size-limit choices. There is no onboarding task picker, suggested queries, or deep-analysis questionnaire.
- This agent does not train, sweep, or predict. List-valued composition columns are valid; do not flatten them solely to look ready for modeling.

## Contributing

See the Discovery catalog `CONTRIBUTING.md` for contribution guidelines.
