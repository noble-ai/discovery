# NobleAI Formulation Analytics Agent

Explores chemical formulation and tabular experimental CSVs on Microsoft Discovery using Foundry Code Interpreter. It profiles data, computes statistics, and writes self-contained Plotly HTML charts. It does not train models.

## Overview

Chemists and materials scientists often need to inspect formulation tables (component IDs and proportions, targets, process covariates) before any modeling run. This agent answers those exploratory questions in chat: schema and missingness, composition views, correlations, simple cleaning, and a short model-readiness narrative.

It is **not** the NobleAI Formulation Modeling Agent. Modeling (Hydra YAML, `run_experiment`, sweeps, predict) stays on that agent. Use this agent for analysis and charts; switch to formulation modeling with the same uploaded CSVs when you are ready to train.

This version is a prompt agent plus Foundry Code Interpreter. It does not ship a container, Noble analytics tools (`create_chart`, sandbox interpreter, reports, memory), suggested queries, or deep-analysis mode.

## Architecture

```
User CSV upload → prompt agent → Foundry Code Interpreter
  → Plotly HTML + optional CSV files
  → PreviewResource / ShareResource
  → markdown takeaways in chat
```

- **Model**: workspace chat deployment (`{{CHAT-MODEL}}`).
- **Compute**: Microsoft-managed Code Interpreter sandbox (pandas and, when installed, Plotly). No Discovery `code_environments` image and no modeling runtime.
- **Charts**: Plotly `fig.write_html(..., include_plotlyjs=True)` so the HTML is self-contained. Discovery chat does not render an inline Plotly widget; the user previews or downloads the HTML file.
- **Tables**: small summaries as markdown in the assistant reply; large results as CSV files in the workspace.
- **Handoff**: if the user asks to train or predict, the agent points them at the formulation modeling agent. It does not author training YAML.

No session memory is stored beyond the conversation and shared files.

## Prerequisites

- Microsoft Discovery workspace with a deployed project
- Azure AI Foundry chat model deployment (`{{CHAT-MODEL}}`)
- Foundry **Code Interpreter** enabled on the agent (declared in `agent.yaml` as `kind: code_interpreter`; confirm it is attached in Foundry / Discovery Studio at deploy time)
- Formulation or tabular CSV uploads (optional second component/chemical database CSV)

Sample formulation CSVs used by the modeling agent (do not need to be copied here):

- `agents/formulation-modeling-agent/tools/ml-core/example-input-files/basic/formulation_sample.csv`
- `agents/formulation-modeling-agent/tools/ml-core/example-input-files/basic/components_sample.csv`

## Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{{CHAT-MODEL}}` | Azure AI Foundry model deployment name | `gpt-4o` |

No container image, ACR, or Discovery tool ARM IDs are required.

If Studio does not attach Code Interpreter from catalog YAML alone, enable it on the prompt agent in the Foundry UI when you deploy.

## Usage

1. Deploy the agent in Discovery Studio and confirm Code Interpreter is available.
2. Upload a formulation CSV (and optional component CSV).
3. Ask an exploratory question. The agent inspects files in Code Interpreter, writes Plotly HTML (and CSVs when needed), then previews and shares those files.

Example prompts (documentation only; this agent does not show suggestion chips):

| Prompt | Input | Expected output |
|--------|--------|-----------------|
| Profile my formulation dataset | formulation CSV | Column types, missingness, list-column alignment, short markdown summary |
| Show composition for the first 20 formulations | formulation CSV | Self-contained `composition_hbar.html` (exploded IDs / Proportions) |
| How does temperature relate to viscosity? | CSV with T and a viscosity column | Plotly scatter HTML plus correlation / range stats |
| What is missing, and is this ready to model viscosity? | formulation CSV, optional component CSV | Missingness notes and readiness *advice*; no YAML and no training |

After a chart is shared, open the HTML file to interact with the Plotly figure.

When you want a trained property model, use **NobleAI Formulation Modeling Agent** with the same files.

## Known Limitations

- Charts are HTML files, not inline Plotly widgets or `{{chart_N}}` gallery tokens.
- Plotly must be importable in the Foundry Code Interpreter sandbox. If `import plotly` fails, the agent reports that and does not fall back to PNG. Treat a missing Plotly install as a reason to add a custom coding-environment image later; this version does not add that image.
- The Code Interpreter package set is Microsoft-managed. Helpers from Noble analytics (deterministic profiler, chart tools, memory) are not installed.
- Code Interpreter stdout may be truncated; important results should be in files plus a short chat summary.
- The planner should invoke only one tool at a time.
- Human-in-the-loop (`GetUserInput`) is enabled for ambiguous questions only; there is no onboarding task picker, suggested queries, or deep-analysis questionnaire.
- This agent does not train, sweep, or predict. List-valued composition columns are valid; do not flatten them solely to look "model-ready."

## Contributing

See the Discovery catalog `CONTRIBUTING.md` for contribution guidelines.
