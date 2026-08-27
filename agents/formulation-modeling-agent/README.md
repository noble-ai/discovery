# NobleAI Formulation Modeling Agent for Microsoft Discovery

Expert agent for training property prediction models on chemical formulation experimental data using [noble-ml-core](https://github.com/nobleai/noble-ml-core).

## Overview

This agent wraps NobleAI's formulation ML stack for the Microsoft Discovery platform. Users upload formulation CSV data; the agent inspects it in the coding environment, authors a Hydra YAML, then runs noble-ml-core actions to train, sweep, and predict.

Supported workflows:

- **Regression**: Predict continuous properties (viscosity, logV, strength, modulus)
- **Classification**: Binary property labels (via `problem_type='binary-classification'`)
- **Data profiling**: Validate formulation schema before training (coding environment helpers)
- **Chemical featurization**: RDKit, Mordred, Morgan, ChemBERTa, Mol2Vec, and more (requires a component table with SMILES; pass its path explicitly)

The container installs noble-ml-core via UV (same pattern as `noble-ml-core/Dockerfile`) with `models-all`, `training`, `chemical-featurizers-all`, and `cpu` extras.

## Prerequisites

- Microsoft Discovery workspace with a deployed project
- Azure Container Registry (ACR) configured
- Chat model deployment (`{{CHAT-MODEL}}`)
- Docker or Discovery Toolbox ACR Tasks for image builds

## Build Docker Image

From `tools/ml-core/`:

```bash
# Stage noble-ml-core into a self-contained build context (~14 MB).
# Required for az acr build (does not support Docker --build-context).
export NOBLE_ML_CORE_PATH=/path/to/noble-ml-core
STAGE=$(./stage-build-context.sh "$NOBLE_ML_CORE_PATH")

# Recommended: build in ACR (avoids local docker push of ~1 GB layers)
az acr login --name <acr-name>
az acr build \
  --registry <acr-name> \
  --image ml-core:latest \
  "$STAGE"

# Local build from staged context
docker build -t ml-core:latest "$STAGE"
```

Override extras (optional):

```bash
docker build -t ml-core:latest \
  --build-arg EXTRAS=models-forest,models-torch,training,chemical-featurizers-all \
  --build-arg INFERENCE_BACKEND=cpu \
  "$STAGE"
```

If the build fails with `Insufficient disk space`, free Docker disk first:

```bash
docker system prune -a --volumes
```

### Push (local build only)

Local `docker push` often fails on this image (~1 GB layers) because Docker Desktop routes uploads through an internal proxy (`192.168.65.1:3128`). A proxy bypass list frequently does not apply to push traffic — disable the proxy entirely and restart Docker Desktop, or use `az acr build` above.

```bash
docker tag ml-core:latest <acr-name>.azurecr.io/ml-core:latest
az acr login --name <acr-name>
docker push <acr-name>.azurecr.io/ml-core:latest
```

## Local Testing

```bash
cd tools/ml-core

# Unit tests (requires noble-ml-core installed locally)
pip install -e /path/to/noble-ml-core[models-forest,models-torch,training,chemical-featurizers-all,cpu] \
  --extra-index-url https://download.pytorch.org/whl/cpu
pytest test_ml_core_utils.py -v

# End-to-end training test
python test_e2e_formulation.py

# Container smoke test
STAGE=$(./stage-build-context.sh "$NOBLE_ML_CORE_PATH")
docker build -t ml-core:latest "$STAGE"
docker run --rm \
  -v "$(pwd)/example-input-files:/input" \
  -v "$(pwd)/_output:/output" \
  ml-core:latest \
  python3 test_e2e_formulation.py
```

## Usage

The agent inspects data and writes `experiment_config.yaml` (or `sweep_config.yaml`) in the coding environment, then calls a tool action with `config_path`.

| Prompt | Input File(s) | Description |
|--------|---------------|-------------|
| Train a model to predict logV from my formulation data | formulation_data.csv (IDs, Proportions, logV) | Author YAML, then `run_experiment` |
| Profile my uploaded formulation dataset | formulation_data.csv | Schema validation and target statistics |
| Train with RDKit fingerprints | formulation_data.csv + components.csv | Chemical featurization from SMILES |
| Sweep XGBoost hyperparameters | formulation_data.csv | Author YAML with a `tune` block, then `run_hyperparameter_sweep` |

## Expected Data Format

### Formulation CSV

| Column | Required | Description |
|--------|----------|-------------|
| `IDs` | Yes | JSON array of component IDs, e.g. `"[240, 777]"` |
| `Proportions` | Yes | JSON array of weights, e.g. `"[0.2, 0.8]"` |
| Target (e.g. `logV`) | Yes | Numeric property to predict |
| Covariates (e.g. `T`) | No | Extra numeric features included automatically |

### Component database (for chemical featurizers)

| Column | Required | Description |
|--------|----------|-------------|
| `IDs` | Yes | Component ID matching formulation CSV (stringified) |
| `SMILES` | Yes | SMILES string for RDKit/Mordred/ChemBERTa featurizers |

See `tools/ml-core/example-input-files/` for sample files.

Example configs:

- `basic/local_experiment_config.yaml` — local one-hot formulation baseline
- `basic/multi_architecture_sweep_config.yaml` — conditional model-architecture sweep
- `viscosity_analytics/rdkit_experiment_config.yaml` — component database + RDKit features
- `viscosity_analytics/component_uniqueness_experiment_config.yaml` — component-level extrapolation split

## Models and Processors

In the coding environment, call `list_models()`, `list_preprocessors()`, `list_postprocessors()`, and `get_component_config(name, type)` to see what is installed and which Hydra fields each component accepts. Chemical featurizers (RDKit, Mordred, Morgan, ChemBERTa, …) are processors — list them via `list_preprocessors()` and place them in the preprocessing `component_pipeline`. Assemble YAML with `build_component_config` / `create_config_from_yaml_string`, then `validate_experiment_config` before saving.

Do not train from scripts. Pass the YAML to the `run_experiment` or `run_hyperparameter_sweep` action. Pass formulation and component-table paths explicitly to `prepare_formulation_data` (`formulation_path`, optional `component_db_path`).

```python
from ml_core_utils import (
    quick_setup, quick_finish, save_final_results,
    list_models, get_component_config, build_component_config,
    build_formulation_data_config,
    create_config_from_yaml_string, validate_experiment_config, save_experiment_config,
)

quick_setup(input_dir="/input", output_dir="/output", work_dir="/workdir")
print(list_models("regression"))
print(list_preprocessors())
print(get_component_config("XGBModel", "model"))
cfg = create_config_from_yaml_string(yaml_text)  # four blocks: data, pipelines, model
cfg.data = build_formulation_data_config(
    formulation_file="formulation_training_ready.pkl",
    component_file_path="ml_core_component_database.csv",
    input_columns=["IDs", "Proportions", "T"],
    target_columns=["logV"],
)
assert validate_experiment_config(cfg)["ok"]
save_experiment_config(cfg, "/workdir/experiment_config.yaml")
save_final_results({"config": "experiment_config.yaml"}, {"experiment_config": "/workdir/experiment_config.yaml"})
quick_finish()
```

## Multi-architecture sweeps

Noble-ml-core can sweep across model or processor classes, not only parameters
within one class. Following
`noble-ml-core/examples/configs/hyperparameter-example.yaml` and
`noble-ml-core/examples/configs/formulation-sweep-base.yaml`, replace `_target_`
with `_class_options` and define the union of candidate-class parameters:

```yaml
model:
  _class_options:
    search_space: choice
    categories: [LinearModel, XGBModel]
  config:
    problem_type: regression
    model_class:
      search_space: choice
      categories: [ElasticNet, Ridge, LinearRegression]
    alpha:
      search_space: loguniform
      lower: 0.001
      upper: 10.0
    n_estimators:
      search_space: choice
      categories: [50, 100, 200]
    max_depth:
      search_space: randint
      lower: 3
      upper: 10

tune:
  resources_per_trial: ${oc.decode:"{cpu:1, gpu:0}"}
  metric: score
  mode: max
  tune_config:
    _target_: ray.tune.TuneConfig
    num_samples: 10
    max_concurrent_trials: 1
    search_alg:
      _target_: ray.tune.search.optuna.OptunaSearch
      metric: ${tune.metric}
      mode: ${tune.mode}
```

`run_sweep` builds a conditional search space, so each sampled architecture only
receives fields supported by its registered config class:

```python
from omegaconf import OmegaConf
from noble_ml_core.training.hyperparameter_tuning.run_sweep import run_sweep

cfg = OmegaConf.load("multi_architecture_sweep_config.yaml")
results = run_sweep(cfg=cfg, output_dir="./sweep_results")
```

In Discovery, author and validate the same YAML in the coding environment, then
use the execute action (which wraps the call above):

```bash
python entrypoint.py --action run_hyperparameter_sweep \
  --input /input --output /output \
  --config-path /input/multi_architecture_sweep_config.yaml
```

The complete formulation example is
`tools/ml-core/example-input-files/basic/multi_architecture_sweep_config.yaml`.
The formulation `preprocessing_pipeline` still requires `FormulationProcessor`.
Processor architecture sweeps use the same `_class_options` structure and may
include `null` to represent no processor.

## File Structure

```
formulation-modeling-agent/
├── agent.yaml
├── metadata.yaml
├── README.md
└── tools/ml-core/
    ├── tool.yaml
    ├── Dockerfile
    ├── ml_core_utils.py
    ├── test_ml_core_utils.py
    ├── test_e2e_formulation.py
    └── example-input-files/
        ├── formulation_sample.csv
        └── components_sample.csv
```

## Architecture

```
User Upload → coding env (inspect + YAML) → ml-core actions (entrypoint.py) → noble-ml-core → metrics/artifacts
```

- **Agent**: Prompt agent that inspects data and authors Hydra YAML in the coding environment, then calls execute-only actions
- **Tool**: One container with a coding environment plus four actions (`run_experiment`, `run_hyperparameter_sweep`, `train_best_from_sweep`, `predict`)
- **ML engine**: noble-ml-core `run_experiment` / `run_sweep` from the authored config

## Tools

The `ml-core` tool exposes a Python coding environment (inspect + config authoring only) and these actions:

| Action | Purpose |
|--------|---------|
| `run_experiment` | Train from `config_path` (required) → metrics, plots, optional artifacts |
| `run_hyperparameter_sweep` | Ray Tune from a YAML that already includes `tune` → `best_config.yaml` |
| `train_best_from_sweep` | Retrain best config with LOCAL artifacts |
| `predict` | Score new rows from saved artifacts |

Local smoke (no Docker), after a YAML exists:

```bash
python entrypoint.py --action run_experiment \
  --input ./example-input-files/basic --output /tmp/ml-out \
  --config-path ./example-input-files/basic/experiment_config.yaml
```

## Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{{CHAT-MODEL}}` | Azure AI Foundry model deployment name | `gpt-4o` |
| `{{mlCoreToolId}}` | ARM resource ID of the deployed ml-core tool | set at deploy time |

## Support

- NobleAI: support@noble.ai
- Issues: https://github.com/nobleai/discovery-catalog/issues

## Known Limitations

- Formulation CSV list columns must be JSON-encoded strings or Python lists (platform-wide format is converted in the coding environment via `prepare_formulation_data`)
- Chemical processors that use SMILES require an explicit `component_db_path` to a table mapped to component IDs
- Training and sweeps require an authored Hydra YAML (`config_path`); the agent must not train from scripts
- Hyperparameter sweeps require Ray/Optuna (included via noble-ml-core `training` extras) and are CPU-heavy
- Full SageMaker-style endpoint deployment is not included; use `predict` on LOCAL artifacts
- Neural models (FCN, ChemBERTa) are CPU-only in this container and may be slow on large datasets

## Contributing

See the Discovery catalog `CONTRIBUTING.md` for contribution guidelines.
