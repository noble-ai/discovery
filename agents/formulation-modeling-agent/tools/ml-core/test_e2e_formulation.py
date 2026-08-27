#!/usr/bin/env python3
"""End-to-end test: author a Hydra config and run_experiment_action."""

import json
import os
import sys
import traceback

print("=" * 60)
print("STEP 0: Verifying imports")
print("=" * 60)

try:
    import numpy as np
    import pandas as pd
    import sklearn
    import noble_ml_core

    print(f"  numpy: {np.__version__}")
    print(f"  pandas: {pd.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")
    print(f"  noble-ml-core: {noble_ml_core.__version__}")

    from ml_core_utils import (
        create_config_from_yaml_string,
        prepare_formulation_data,
        quick_finish,
        quick_setup,
        run_experiment_action,
        save_experiment_config,
        save_final_results,
        validate_experiment_config,
    )

    print("  ml_core_utils: OK")
    print("ALL IMPORTS SUCCESSFUL")
except ImportError as exc:
    print(f"IMPORT ERROR: {exc}")
    traceback.print_exc()
    sys.exit(1)

quick_setup(input_dir="/input", output_dir="/output", work_dir="/workdir")

results = {}
output_files = {}

try:
    sample_dir = "/input"
    if not os.path.exists(os.path.join(sample_dir, "formulation_sample.csv")):
        sample_dir = os.path.join(os.path.dirname(__file__), "example-input-files", "basic")

    print("\n" + "=" * 60)
    print("STEP 1: Prepare formulation data")
    print("=" * 60)
    prep = prepare_formulation_data(
        sample_dir,
        "/workdir",
        formulation_path=os.path.join(sample_dir, "formulation_sample.csv"),
        component_db_path=os.path.join(sample_dir, "components_sample.csv")
        if os.path.isfile(os.path.join(sample_dir, "components_sample.csv"))
        else None,
        target_columns=["logV"],
    )
    pickle_name = os.path.basename(prep["training_pickle_path"])
    print(f"Prepared {pickle_name}; targets={prep.get('target_columns')}")

    print("\n" + "=" * 60)
    print("STEP 2: Author experiment_config.yaml")
    print("=" * 60)
    yaml_text = f"""
data:
  dataloader:
    _target_: noble_ml_core.data.formulation_dataloader.LocalFormulationDataLoader
  config:
    _target_: noble_ml_core.data.formulation_dataloader.LocalFormulationDataLoaderHyperparameterSet
    file_path: {pickle_name}
    input_columns:
      - IDs
      - Proportions
      - T
    target_columns:
      - logV
    problem_type: regression
    split_method: train_test_val
    test_size: 0.2
    random_state: 42
preprocessing_pipeline:
  _target_: noble_ml_core.processors.processor_pipelines.SequentialProcessorPipeline
  processors:
    - _target_: noble_ml_core.processors.formulation_processors.FormulationProcessor
      config:
        _target_: noble_ml_core.processors.formulation_processors.FormulationProcessorHyperparameterSet
        keep_original_columns: true
        formulation_column: IDs
        quantity_column: Proportions
        input_columns:
          - IDs
          - Proportions
        component_pipeline:
          _target_: noble_ml_core.processors.processor_pipelines.SequentialProcessorPipeline
          processors:
            - _target_: noble_ml_core.processors.basic_processors.OneHotEncoderProcessor
              config:
                _target_: noble_ml_core.processors.basic_processors.OneHotEncoderProcessorHyperparameterSet
                input_columns:
                  - IDs
        aggregation_function:
          _target_: noble_ml_core.processors.aggregators.MeanAggregationFunction
          mean_type: arithmetic
    - _target_: noble_ml_core.processors.feature_selection.FeatureSelectorProcessor
      config:
        _target_: noble_ml_core.processors.feature_selection.FeatureSelectorProcessorHyperparameterSet
        dropped_columns:
          - IDs
          - Proportions
    - _target_: noble_ml_core.processors.basic_processors.ScalerProcessor
      config:
        _target_: noble_ml_core.processors.basic_processors.ScalerProcessorHyperparameterSet
        type: standard
postprocessing_pipeline:
  _target_: noble_ml_core.processors.processor_pipelines.InvertibleSequentialProcessorPipeline
  processors:
    - _target_: noble_ml_core.processors.basic_processors.ScalerProcessor
      config:
        _target_: noble_ml_core.processors.basic_processors.ScalerProcessorHyperparameterSet
        type: standard
model:
  _target_: noble_ml_core.models.xgb_model.XGBModel
  config:
    _target_: noble_ml_core.models.xgb_model.XGBModelHyperparameterSet
    problem_type: regression
"""
    cfg = create_config_from_yaml_string(yaml_text)
    check = validate_experiment_config(cfg)
    if not check["ok"]:
        raise ValueError(check["errors"])
    config_path = save_experiment_config(cfg, "/workdir/experiment_config.yaml")

    print("\n" + "=" * 60)
    print("STEP 3: run_experiment_action")
    print("=" * 60)
    trained = run_experiment_action(
        "/workdir",
        "/output",
        config_path=config_path,
        save_artifacts=True,
        generate_plots=True,
    )
    results["metrics"] = trained.get("metrics", {})
    results["config_path"] = trained.get("config_path")
    output_files.update(trained.get("output_files") or {})
    print("\nMetrics:", json.dumps(results["metrics"], indent=2, default=str))

except Exception as exc:
    print(f"ERROR: {exc}")
    traceback.print_exc()
    results["error"] = str(exc)
finally:
    save_final_results(results, output_files)
    quick_finish()

print("\nE2E TEST COMPLETE")
