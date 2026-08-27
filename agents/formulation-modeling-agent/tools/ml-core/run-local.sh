#!/usr/bin/env bash
# Layer 1 local test: run ml_core_utils against formulation CSVs without Docker.
#
# Usage:
#   ./run-local.sh
#   ./run-local.sh /path/to/data/dir
#   ./run-local.sh --preprocessor OneHotEncoderProcessor --model XGBModel
#   ./run-local.sh --max-rows 500
#
# Defaults to example-input-files/viscosity_analytics/ with formulation_data.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/example-input-files/viscosity_analytics"
FORMULATION_CSV="formulation_data.csv"
COMPONENT_CSV="chemical_db.csv"
TARGET="logV"
MODEL_TYPE="XGBModel"
PREPROCESSOR="OneHotEncoderProcessor"
MAX_ROWS=""
PYTHON="${PYTHON:-python3}"

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --target) TARGET="$2"; shift 2 ;;
    --model) MODEL_TYPE="$2"; shift 2 ;;
    --preprocessor|--featurizer) PREPROCESSOR="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --*)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
    *)
      DATA_DIR="$1"
      shift
      ;;
  esac
done

FORMULATION_PATH="${DATA_DIR}/${FORMULATION_CSV}"
COMPONENT_PATH="${DATA_DIR}/${COMPONENT_CSV}"
OUTPUT_DIR="${DATA_DIR}/_output"
WORK_DIR="${DATA_DIR}/_workdir"

if [[ ! -f "$FORMULATION_PATH" ]]; then
  echo "ERROR: formulation CSV not found: $FORMULATION_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WORK_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "=== ml-core Layer 1 local run ==="
echo "Data dir:       $DATA_DIR"
echo "Formulation:    $FORMULATION_PATH"
echo "Target:         $TARGET"
echo "Model:          $MODEL_TYPE"
echo "Preprocessor:   $PREPROCESSOR"
echo "Output:         $OUTPUT_DIR"
echo "Python:         $PYTHON"
[[ -n "$MAX_ROWS" ]] && echo "Max rows:       $MAX_ROWS"
echo

exec "$PYTHON" - "$FORMULATION_PATH" "$COMPONENT_PATH" "$OUTPUT_DIR" "$WORK_DIR" \
  "$TARGET" "$MODEL_TYPE" "$PREPROCESSOR" "$MAX_ROWS" <<'PY'
import json
import os
import sys
import traceback

import pandas as pd
from omegaconf import OmegaConf

(
    formulation_path,
    component_path,
    output_dir,
    work_dir,
    target,
    model_type,
    preprocessor,
    max_rows,
) = sys.argv[1:9]

max_rows = int(max_rows) if max_rows else None
alias_models = {"xgb": "XGBModel", "linear": "LinearModel", "tabpfn": "TabPFNModel"}
alias_processors = {"onehot": "OneHotEncoderProcessor", "rdkit": "RDKitFeaturizer"}
model_type = alias_models.get(model_type.lower(), model_type)
preprocessor = alias_processors.get(preprocessor.lower(), preprocessor)

from ml_core_utils import (
    build_component_config,
    prepare_formulation_data,
    quick_finish,
    quick_setup,
    run_experiment_action,
    save_experiment_config,
    save_final_results,
    validate_experiment_config,
)

input_dir = os.path.dirname(formulation_path)
quick_setup(input_dir=input_dir, output_dir=output_dir, work_dir=work_dir)

results = {}
output_files = {}
exit_code = 0

try:
    prep = prepare_formulation_data(
        input_dir,
        work_dir,
        formulation_path=formulation_path,
        component_db_path=component_path if os.path.isfile(component_path) else None,
        target_columns=[target],
    )
    pickle_path = prep["training_pickle_path"]
    if max_rows:
        pd.read_pickle(pickle_path).head(max_rows).to_pickle(pickle_path)

    pickle_name = os.path.basename(pickle_path)
    input_cols = prep.get("input_columns") or ["IDs", "Proportions"]
    component_name = (
        os.path.basename(prep["aligned_component_db_path"])
        if prep.get("aligned_component_db_path")
        else None
    )

    if preprocessor == "RDKitFeaturizer":
        encoder = build_component_config(
            preprocessor, "processor", {"smiles_column": "SMILES"}
        )
        dropper = {
            "_target_": "noble_ml_core.processors.feature_selection.FeatureSelectorProcessor",
            "config": {
                "_target_": (
                    "noble_ml_core.processors.feature_selection."
                    "FeatureSelectorProcessorHyperparameterSet"
                ),
                "dropped_columns": ["name", "salt", "index", "IDs"],
            },
        }
        component_processors = [dropper, encoder]
    else:
        encoder = build_component_config(
            preprocessor, "processor", {"input_columns": ["IDs"]}
        )
        component_processors = [encoder]

    model = build_component_config(model_type, "model", {"problem_type": "regression"})
    data_config = {
        "_target_": (
            "noble_ml_core.data.formulation_dataloader."
            "LocalFormulationDataLoaderHyperparameterSet"
        ),
        "file_path": pickle_name,
        "input_columns": input_cols,
        "target_columns": [target],
        "problem_type": "regression",
        "split_method": "train_test_val",
        "test_size": 0.2,
        "random_state": 42,
    }
    if component_name:
        data_config["component_file_path"] = component_name

    cfg = OmegaConf.create(
        {
            "data": {
                "dataloader": {
                    "_target_": (
                        "noble_ml_core.data.formulation_dataloader."
                        "LocalFormulationDataLoader"
                    ),
                },
                "config": data_config,
            },
            "preprocessing_pipeline": {
                "_target_": (
                    "noble_ml_core.processors.processor_pipelines."
                    "SequentialProcessorPipeline"
                ),
                "processors": [
                    {
                        "_target_": (
                            "noble_ml_core.processors.formulation_processors."
                            "FormulationProcessor"
                        ),
                        "config": {
                            "_target_": (
                                "noble_ml_core.processors.formulation_processors."
                                "FormulationProcessorHyperparameterSet"
                            ),
                            "keep_original_columns": True,
                            "formulation_column": "IDs",
                            "quantity_column": "Proportions",
                            "input_columns": ["IDs", "Proportions"],
                            "component_pipeline": {
                                "_target_": (
                                    "noble_ml_core.processors.processor_pipelines."
                                    "SequentialProcessorPipeline"
                                ),
                                "processors": component_processors,
                            },
                            "aggregation_function": {
                                "_target_": (
                                    "noble_ml_core.processors.aggregators."
                                    "MeanAggregationFunction"
                                ),
                                "mean_type": "arithmetic",
                            },
                        },
                    },
                    {
                        "_target_": (
                            "noble_ml_core.processors.feature_selection."
                            "FeatureSelectorProcessor"
                        ),
                        "config": {
                            "_target_": (
                                "noble_ml_core.processors.feature_selection."
                                "FeatureSelectorProcessorHyperparameterSet"
                            ),
                            "dropped_columns": ["IDs", "Proportions"],
                        },
                    },
                    {
                        "_target_": (
                            "noble_ml_core.processors.basic_processors.ScalerProcessor"
                        ),
                        "config": {
                            "_target_": (
                                "noble_ml_core.processors.basic_processors."
                                "ScalerProcessorHyperparameterSet"
                            ),
                            "type": "standard",
                        },
                    },
                ],
            },
            "postprocessing_pipeline": {
                "_target_": (
                    "noble_ml_core.processors.processor_pipelines."
                    "InvertibleSequentialProcessorPipeline"
                ),
                "processors": [
                    {
                        "_target_": (
                            "noble_ml_core.processors.basic_processors.ScalerProcessor"
                        ),
                        "config": {
                            "_target_": (
                                "noble_ml_core.processors.basic_processors."
                                "ScalerProcessorHyperparameterSet"
                            ),
                            "type": "standard",
                        },
                    },
                ],
            },
            "model": model,
        }
    )
    check = validate_experiment_config(cfg)
    if not check["ok"]:
        raise ValueError(check["errors"])
    config_path = save_experiment_config(
        cfg, os.path.join(work_dir, "experiment_config.yaml")
    )
    trained = run_experiment_action(
        work_dir,
        output_dir,
        config_path=config_path,
        save_artifacts=True,
        generate_plots=True,
    )
    results["metrics"] = trained.get("metrics", {})
    results["model"] = model_type
    results["preprocessor"] = preprocessor
    output_files.update(trained.get("output_files") or {})
    print("\nMetrics:")
    print(json.dumps(results["metrics"], indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    traceback.print_exc()
    results["error"] = str(exc)
    exit_code = 1
finally:
    save_final_results(results, output_files)
    quick_finish()
    print(f"\nWrote results to {output_dir}/final_results.json")

sys.exit(exit_code)
PY
