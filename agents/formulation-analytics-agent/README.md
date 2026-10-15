# NobleAI Formulation Analytics Agent

Explores chemical formulation and tabular experimental CSVs on Microsoft Discovery. A Python coding environment profiles data, computes statistics, and writes self-contained Plotly HTML charts. It does not train models.

## Overview

Chemists and materials scientists often need to inspect formulation tables (component IDs and proportions, targets, process covariates) before any modeling run. This agent answers those exploratory questions in chat: schema and missingness, composition views, correlations, simple cleaning, and a short model-readiness narrative.

It is **not** the NobleAI Formulation Modeling Agent. Modeling (Hydra YAML, training, sweeps, predict) stays on that agent. Use this agent for analysis and charts; switch to formulation modeling with the same uploaded CSVs when you are ready to train.

This version is a prompt agent plus one coding-environment container. It does not use Foundry Code Interpreter, chart or table widgets, reports, memory, suggested queries, or deep-analysis mode.

## Architecture

```
Linked CSV storage asset
  → mount parent at /input (optional prior output at /state)
  → one Python script in the analytics coding environment
  → /output/charts/*.html, /output/tables/*.csv, /output/state/*
  → PreviewResource / ShareResource
  → markdown takeaways in chat
```

- **Model**: workspace chat deployment (`{{CHAT-MODEL}}`).
- **Compute**: container image `formulation-analytics` (pandas, scipy, scikit-learn, statsmodels, pyarrow, plotly). Discovery starts the container and mounts files. There is no nested sandbox and no persistent Python kernel.
- **Charts**: `write_plotly_html` saves self-contained HTML (`include_plotlyjs=True`). Discovery chat does not render an inline Plotly widget; the user previews or downloads the HTML file.
- **Tables**: small summaries as markdown in the assistant reply; large results as CSV files under `/output/tables`.
- **State**: variables die when the job exits. Reusable tables are Parquet under `/output/state`. A later job remounts that output at `/state` only when it needs them.
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

Build and register `tools/analytics-runtime` before the agent can run scripts. The catalog deployer builds this image when the agent is deployed from the repo.

## Usage

1. Deploy the agent and the analytics-runtime tool.
2. Link a formulation CSV (and optional component CSV) as a storage asset on the session.
3. Ask an exploratory question. The agent mounts the parent directory at `/input`, runs one script, writes Plotly HTML (and CSVs when needed), then previews and shares those files.

Example prompts (documentation only; this agent does not show suggestion chips):

| Prompt | Input | Expected output |
|--------|--------|-----------------|
| Profile my formulation dataset | formulation CSV | Column types, missingness, list-column alignment, short markdown summary |
| Show composition for the first 20 formulations | formulation CSV | Self-contained `/output/charts/composition_hbar.html` |
| How does temperature relate to viscosity? | CSV with T and a viscosity column | Plotly scatter HTML plus correlation / range stats |
| What is missing, and is this ready to model viscosity? | formulation CSV, optional component CSV | Missingness notes and readiness *advice*; no YAML and no training |

After a chart is shared, open the HTML file to interact with the Plotly figure.

When you want a trained property model, use **NobleAI Formulation Modeling Agent** with the same files.

## Tools

### analytics-runtime

Coding environment only. No execute actions.

| Path | Role |
|------|------|
| `/input` | Parent directory of uploaded CSVs |
| `/state` | Optional remount of a previous job's `/output` |
| `/output` | Charts, CSV tables, parquet state, and logs |

Helpers in the image: `parse_list_columns`, `write_plotly_html`.

The generated script is run with:

```text
python3 -u "/{{scriptName}}" > {{output_directory}}/logs.out 2>&1; tail -n 100 {{output_directory}}/logs.out
```

Stdout in the agent reply is the last 100 log lines. Put results the user should keep in files under `/output`.

## Known Limitations

- Charts are HTML files, not inline Plotly widgets.
- Each coding job is a new process. Assigning a variable does not make it available on the next call. Persist Parquet and JSON under `/output/state` and remount that output at `/state`.
- One tool call at a time. Prefer one script per question so inspect, transform, and plots do not each pay for a new job.
- Logs returned to the agent are truncated to the last 100 lines.
- Human-in-the-loop (`GetUserInput`) is enabled for ambiguous questions only. There is no onboarding task picker, suggested queries, or deep-analysis questionnaire.
- This agent does not train, sweep, or predict. List-valued composition columns are valid; do not flatten them solely to look ready for modeling.

## Contributing

See the Discovery catalog `CONTRIBUTING.md` for contribution guidelines.
