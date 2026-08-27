#!/usr/bin/env python3
"""Unit tests for ml_core_utils.py."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))
from ml_core_utils import (
    DEFAULT_FORMULATION_COLUMN,
    DEFAULT_QUANTITY_COLUMN,
    DEFAULT_SMILES_COLUMN,
    _extract_flat_metrics,
    _parse_list_value,
    _rebase_config_paths,
    align_component_database,
    assess_model_readiness,
    build_component_config,
    build_formulation_data_config,
    convert_platform_to_lists,
    create_config_from_yaml_string,
    detect_data_format,
    get_component_config,
    list_models,
    list_postprocessors,
    list_preprocessors,
    list_split_method_config_parameters,
    list_split_methods,
    load_component_database,
    load_formulation_csv,
    prepare_formulation_data,
    prepare_formulation_inputs,
    prepare_training_file,
    save_experiment_config,
    save_final_results,
    train_best_from_sweep,
    validate_experiment_config,
    validate_formulation_schema,
)


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "formulation.csv"
    df = pd.DataFrame(
        {
            DEFAULT_FORMULATION_COLUMN: ["[1, 2]", "[2, 3]", "[1, 3]"],
            DEFAULT_QUANTITY_COLUMN: ["[0.5, 0.5]", "[0.4, 0.6]", "[0.7, 0.3]"],
            "T": [298.15, 303.15, 308.15],
            "logV": [-0.4, -0.3, -0.2],
        }
    )
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def platform_csv(tmp_path):
    path = tmp_path / "platform.csv"
    pd.DataFrame(
        {
            "component_0_identifier": [1, 2],
            "component_0_amount": [0.5, 0.6],
            "component_1_identifier": [2, 3],
            "component_1_amount": [0.5, 0.4],
            "T": [298.0, 300.0],
            "logV": [-0.4, -0.3],
        }
    ).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def components_csv(tmp_path):
    path = tmp_path / "components.csv"
    pd.DataFrame(
        {
            DEFAULT_FORMULATION_COLUMN: ["1", "2", "3"],
            DEFAULT_SMILES_COLUMN: ["CCO", "CC(=O)O", "c1ccccc1"],
        }
    ).to_csv(path, index=False)
    return str(path)


class TestDefaults:
    def test_column_defaults_match_dataloader(self):
        from noble_ml_core.data.formulation_dataloader import (
            LocalFormulationDataLoaderHyperparameterSet,
        )

        defaults = LocalFormulationDataLoaderHyperparameterSet()
        assert DEFAULT_FORMULATION_COLUMN == defaults.id_column
        assert DEFAULT_QUANTITY_COLUMN == defaults.proportions_column
        assert DEFAULT_SMILES_COLUMN == "SMILES"


class TestParseListValue:
    def test_json_string(self):
        assert _parse_list_value("[1, 2, 3]") == [1, 2, 3]

    def test_python_list(self):
        assert _parse_list_value([1, 2]) == [1, 2]


class TestPhase1DataPrep:
    def test_detect_list_format(self, sample_csv):
        raw = pd.read_csv(sample_csv)
        info = detect_data_format(raw)
        assert info["format"] == "list"
        assert info["has_list_columns"] is True

    def test_detect_platform_wide_format(self):
        raw = pd.DataFrame(
            {
                "component_0_identifier": [1, 2],
                "component_0_amount": [0.5, 0.6],
                "component_1_identifier": [2, 3],
                "component_1_amount": [0.5, 0.4],
                "target": [1.0, 2.0],
            }
        )
        info = detect_data_format(raw)
        assert info["format"] == "platform_wide"
        assert info["has_platform_columns"] is True

    def test_convert_platform_to_lists(self, platform_csv):
        raw = pd.read_csv(platform_csv)
        converted = convert_platform_to_lists(raw)
        assert DEFAULT_FORMULATION_COLUMN in converted.columns
        assert DEFAULT_QUANTITY_COLUMN in converted.columns
        assert isinstance(converted[DEFAULT_FORMULATION_COLUMN].iloc[0], list)
        assert abs(sum(converted[DEFAULT_QUANTITY_COLUMN].iloc[0]) - 1.0) < 1e-6
        assert "T" in converted.columns

    def test_align_component_database(self):
        raw = pd.DataFrame(
            {
                "id": [1, 2],
                DEFAULT_SMILES_COLUMN: ["CCO", "CC"],
                "component_name": ["a", "b"],
            }
        )
        aligned = align_component_database(raw)
        assert DEFAULT_FORMULATION_COLUMN in aligned.columns
        assert "component_name" not in aligned.columns
        assert aligned[DEFAULT_FORMULATION_COLUMN].tolist() == ["1", "2"]

    def test_assess_model_readiness(self, sample_csv):
        df = load_formulation_csv(sample_csv, target_columns=["logV"])
        readiness = assess_model_readiness(df, candidate_targets=["logV"])
        assert readiness["verdict"] in {"ready", "conditionally_ready", "not_ready"}
        assert "logV" in readiness["target_candidates"]
        assert "recommended_model_type" not in readiness
        assert "recommended_preprocessor" not in readiness
        assert "covariates" in readiness
        assert readiness["has_component_smiles"] is False

    def test_prepare_formulation_inputs_list_format(self, sample_csv, tmp_path):
        prep = prepare_formulation_inputs(
            sample_csv,
            output_dir=str(tmp_path),
            target_columns=["logV"],
        )
        assert prep["format_detection"]["format"] == "list"
        assert prep["readiness"] is not None
        assert prep["training_pickle_path"] is not None
        assert os.path.exists(prep["training_pickle_path"])

    def test_prepare_formulation_data_platform(self, platform_csv, tmp_path):
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        dest = in_dir / "platform_wide.csv"
        dest.write_text(open(platform_csv).read())
        prep = prepare_formulation_data(
            str(in_dir),
            str(out_dir),
            formulation_path=str(dest),
            target_columns=["logV"],
        )
        assert prep["converted"] is True
        assert prep["training_pickle_path"]
        assert os.path.exists(prep["list_format_csv_path"])

    def test_prepare_formulation_data_requires_explicit_paths(self, sample_csv, tmp_path):
        in_dir = tmp_path / "in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        named = in_dir / "viscosity_runs.csv"
        named.write_text(open(sample_csv).read())
        (in_dir / "chemical_db.csv").write_text("IDs,SMILES\n1,CCO\n2,CC\n3,c1ccccc1\n")
        with pytest.raises(TypeError):
            prepare_formulation_data(str(in_dir), str(out_dir), target_columns=["logV"])
        without_db = prepare_formulation_data(
            str(in_dir),
            str(out_dir),
            formulation_path="viscosity_runs.csv",
            target_columns=["logV"],
        )
        assert without_db["aligned_component_db_path"] is None
        with_db = prepare_formulation_data(
            str(in_dir),
            str(tmp_path / "out2"),
            formulation_path="viscosity_runs.csv",
            component_db_path="chemical_db.csv",
            target_columns=["logV"],
        )
        assert with_db["aligned_component_db_path"]
        assert os.path.exists(with_db["aligned_component_db_path"])


class TestLoadFormulationCsv:
    def test_load_basic(self, sample_csv):
        df = load_formulation_csv(sample_csv, target_columns=["logV"])
        assert len(df) == 3
        assert isinstance(df[DEFAULT_FORMULATION_COLUMN].iloc[0], list)
        assert isinstance(df[DEFAULT_QUANTITY_COLUMN].iloc[0], list)

    def test_missing_formulation_column(self, sample_csv):
        with pytest.raises(ValueError, match="Formulation column"):
            load_formulation_csv(sample_csv, formulation_column="missing")


class TestValidation:
    def test_validate_schema(self, sample_csv):
        df = load_formulation_csv(sample_csv, target_columns=["logV"])
        summary = validate_formulation_schema(df)
        assert summary["n_rows"] == 3
        assert summary["n_unique_components"] >= 3


class TestPrepareTrainingFile:
    def test_writes_pickle(self, sample_csv, tmp_path):
        df = load_formulation_csv(sample_csv, target_columns=["logV"])
        output = tmp_path / "train.pkl"
        path, input_cols, target_cols = prepare_training_file(
            df, output_path=str(output), target_columns=["logV"]
        )
        assert os.path.exists(path)
        assert DEFAULT_FORMULATION_COLUMN in input_cols
        assert target_cols == ["logV"]


class TestSaveFinalResults:
    def test_writes_json(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        os.environ["OUTPUT_DIR"] = str(out_dir)
        import ml_core_utils

        ml_core_utils.OUTPUT_DIR = str(out_dir)
        save_final_results({"score": 0.9}, {"plot": "parity_plot.png"})
        result_path = out_dir / "final_results.json"
        assert result_path.exists()
        payload = json.loads(result_path.read_text())
        assert payload["status"] == "completed"
        assert payload["summary"]["score"] == 0.9


class TestRegistryHelpers:
    def test_list_models_regression(self):
        models = list_models(problem_type="regression")
        assert isinstance(models, dict)
        assert "XGBModel" in models or "LinearModel" in models

    def test_list_preprocessors_includes_onehot(self):
        processors = list_preprocessors()
        assert "OneHotEncoderProcessor" in processors

    def test_list_preprocessors_includes_chemical_featurizers(self):
        processors = list_preprocessors()
        from noble_ml_core import GLOBAL_FEATURIZER_REGISTRY

        for name in GLOBAL_FEATURIZER_REGISTRY.list_components():
            if name == "FeaturizerComponent":
                continue
            assert name in processors

    def test_build_rdkit_as_processor(self):
        pytest.importorskip("rdkit")
        cfg = build_component_config(
            "RDKitFeaturizer",
            "processor",
            {"smiles_column": "SMILES"},
        )
        assert "RDKitFeaturizer" in cfg["_target_"]
        assert cfg["config"].get("smiles_column") == "SMILES"

    def test_list_postprocessors_and_preprocessors(self):
        post = list_postprocessors()
        pre = list_preprocessors()
        assert isinstance(post, dict)
        assert isinstance(pre, dict)
        assert "ScalerProcessor" in post
        assert "OneHotEncoderProcessor" in pre

    def test_get_component_config_model(self):
        schema = get_component_config("XGBModel", "model")
        assert "problem_type" in schema
        assert "type" in schema["problem_type"]

    def test_build_component_config_model(self):
        cfg = build_component_config(
            "XGBModel",
            "model",
            {"problem_type": "regression"},
        )
        assert cfg["_target_"].endswith("XGBModel")
        assert cfg["config"]["_target_"]
        assert cfg["config"].get("problem_type") == "regression"

    def test_build_component_config_processor(self):
        cfg = build_component_config("OneHotEncoderProcessor", "processor")
        assert "OneHotEncoderProcessor" in cfg["_target_"]
        assert "config" in cfg


class TestConfigHelpers:
    def test_build_local_formulation_data_config(self):
        data = build_formulation_data_config(
            "/workdir/custom_formulations.pkl",
            ["logV"],
            component_file_path="/workdir/custom_components.csv",
            input_columns=["IDs", "Proportions", "T"],
        )
        assert data["dataloader"]["_target_"].endswith(
            "LocalFormulationDataLoader"
        )
        assert data["config"]["file_path"] == "custom_formulations.pkl"
        assert data["config"]["component_file_path"] == "custom_components.csv"
        assert data["config"]["input_columns"] == ["IDs", "Proportions", "T"]
        assert data["config"]["target_columns"] == ["logV"]

    def test_build_vip_formulation_data_config(self):
        data = build_formulation_data_config(
            "wide_formulations.csv",
            ["viscosity"],
            loader_type="vip",
            id_column="components",
            proportions_column="fractions",
            input_columns=["components", "fractions", "temperature"],
            vip_id_column="id",
            component_db_drop_columns=["name"],
        )
        assert data["dataloader"]["_target_"].endswith(
            "VipFormulationDataLoader"
        )
        assert data["config"]["_target_"].endswith(
            "VipFormulationDataLoaderHyperparameterSet"
        )
        assert data["config"]["vip_id_column"] == "id"
        assert data["config"]["component_db_drop_columns"] == ["name"]

    def test_build_formulation_data_config_validates_inputs(self):
        with pytest.raises(ValueError, match="formulation_file is required"):
            build_formulation_data_config("", ["target"])
        with pytest.raises(ValueError, match="target_columns"):
            build_formulation_data_config("data.csv", [])
        with pytest.raises(ValueError, match="must include"):
            build_formulation_data_config(
                "data.csv",
                ["target"],
                input_columns=["temperature"],
            )
        with pytest.raises(ValueError, match="Unknown loader_type"):
            build_formulation_data_config(
                "data.csv",
                ["target"],
                loader_type="wandb",
            )
        with pytest.raises(ValueError, match="Unknown split_method"):
            build_formulation_data_config(
                "data.csv",
                ["target"],
                split_method="randomish",
            )

    def test_split_method_context_helpers(self):
        methods = list_split_methods()
        assert "train_test_val" in methods
        assert "kfold_cross_val" in methods
        assert methods["train_test_val"]["signature"].startswith("(")
        assert "docstring" in methods["train_test_val"]
        assert "num_folds" in list_split_method_config_parameters(
            "kfold_cross_val"
        )
        assert "X_train" not in list_split_method_config_parameters(
            "kfold_cross_val"
        )

    def test_split_method_context_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown split method"):
            list_split_method_config_parameters("not_a_split")

    def _minimal_cfg(self, file_path="formulation_training_ready.pkl"):
        return create_config_from_yaml_string(
            f"""
data:
  dataloader:
    _target_: noble_ml_core.data.formulation_dataloader.LocalFormulationDataLoader
  config:
    _target_: noble_ml_core.data.formulation_dataloader.LocalFormulationDataLoaderHyperparameterSet
    file_path: {file_path}
    target_columns: [logV]
preprocessing_pipeline:
  _target_: noble_ml_core.processors.processor_pipelines.SequentialProcessorPipeline
  processors: []
postprocessing_pipeline:
  _target_: noble_ml_core.processors.processor_pipelines.InvertibleSequentialProcessorPipeline
  processors: []
model:
  _target_: noble_ml_core.models.xgb_model.XGBModel
  config:
    _target_: noble_ml_core.models.xgb_model.XGBModelHyperparameterSet
    problem_type: regression
"""
        )

    def test_validate_experiment_config_ok(self):
        check = validate_experiment_config(self._minimal_cfg())
        assert check["ok"] is True
        assert check["has_tune"] is False

    def test_validate_experiment_config_missing_keys(self):
        check = validate_experiment_config({"data": {}})
        assert check["ok"] is False
        assert any("Missing required keys" in e for e in check["errors"])

    def test_validate_experiment_config_checks_targets_and_split_method(self):
        cfg = self._minimal_cfg()
        cfg.data.config.target_columns = []
        cfg.data.config.split_method = "randomish"
        check = validate_experiment_config(cfg)
        assert check["ok"] is False
        assert any("target_columns" in error for error in check["errors"])
        assert any("split_method" in error for error in check["errors"])

    def test_validate_sweep_requires_tune(self):
        check = validate_experiment_config(self._minimal_cfg(), require_tune=True)
        assert check["ok"] is False
        assert any("tune" in e for e in check["errors"])

    def test_save_experiment_config(self, tmp_path):
        path = tmp_path / "experiment_config.yaml"
        save_experiment_config(self._minimal_cfg(), str(path))
        assert path.is_file()

    def test_rebase_config_paths_uses_input_basename(self, tmp_path):
        old_dir = tmp_path / "old"
        new_in = tmp_path / "input"
        new_out = tmp_path / "output"
        old_dir.mkdir()
        new_in.mkdir()
        new_out.mkdir()
        stale = old_dir / "formulation_training_ready.pkl"
        remounted = new_in / "formulation_training_ready.pkl"
        stale.write_text("stale")
        remounted.write_text("fresh")
        cfg = self._minimal_cfg(file_path=str(stale))
        rebased = _rebase_config_paths(cfg, str(new_in), str(new_out))
        assert rebased.data.config.file_path == str(remounted.resolve())
        assert "model_artifacts" in str(rebased.artifact_logging.config.base_path)

    def test_rebase_config_paths_rejects_missing_data(self, tmp_path):
        cfg = self._minimal_cfg(file_path="missing.pkl")
        with pytest.raises(FileNotFoundError, match="Could not resolve data file"):
            _rebase_config_paths(cfg, str(tmp_path), str(tmp_path / "output"))

    def test_flat_metrics_preserve_top_level_score(self):
        metrics = {
            "ensemble": {"r2": 0.8, "rmse": 0.2},
            "score": 0.75,
        }
        assert _extract_flat_metrics(metrics) == {
            "r2": 0.8,
            "rmse": 0.2,
            "score": 0.75,
        }

    def test_train_best_rebases_data_from_input_root(self, tmp_path):
        from unittest.mock import patch

        input_dir = tmp_path / "input"
        sweep_dir = input_dir / "sweep"
        output_dir = tmp_path / "output"
        sweep_dir.mkdir(parents=True)
        output_dir.mkdir()
        data_path = input_dir / "formulation_training_ready.pkl"
        data_path.write_bytes(b"placeholder")
        cfg = self._minimal_cfg(file_path="/old/formulation_training_ready.pkl")
        cfg.tune = {}
        OmegaConf.save(cfg, sweep_dir / "best_config.yaml")

        with patch("ml_core_utils._run_config_experiment") as run_mock:
            run_mock.return_value = ({}, {"score": 0.9}, {})
            train_best_from_sweep(
                str(sweep_dir),
                str(output_dir),
                input_directory=str(input_dir),
                generate_plots=False,
            )

        called_cfg = run_mock.call_args.args[0]
        assert called_cfg.data.config.file_path == str(data_path.resolve())
        assert "tune" not in called_cfg


class TestComponentDatabase:
    def test_load_component_database(self, components_csv):
        db = load_component_database(components_csv)
        assert len(db) == 3
        assert DEFAULT_SMILES_COLUMN in db.columns
        assert db[DEFAULT_FORMULATION_COLUMN].dtype == object


class TestEntrypointParser:
    def test_action_choices(self):
        from entrypoint import ACTIONS, build_parser

        parser = build_parser()
        expected = {
            "run_experiment",
            "run_hyperparameter_sweep",
            "train_best_from_sweep",
            "predict",
        }
        assert set(ACTIONS) == expected
        args = parser.parse_args(
            [
                "--action",
                "run_experiment",
                "--input",
                "/i",
                "--output",
                "/o",
                "--config-path",
                "/i/experiment_config.yaml",
            ]
        )
        assert args.action == "run_experiment"
        assert args.config_path == "/i/experiment_config.yaml"

    def test_run_experiment_requires_config_path(self):
        from entrypoint import action_run_experiment, build_parser

        args = build_parser().parse_args(
            ["--action", "run_experiment", "--input", "/i", "--output", "/o"]
        )
        with pytest.raises(ValueError, match="config_path is required"):
            action_run_experiment(args)

    def test_sweep_requires_config_path(self):
        from entrypoint import action_run_hyperparameter_sweep, build_parser

        args = build_parser().parse_args(
            [
                "--action",
                "run_hyperparameter_sweep",
                "--input",
                "/i",
                "--output",
                "/o",
            ]
        )
        with pytest.raises(ValueError, match="config_path is required"):
            action_run_hyperparameter_sweep(args)

    def test_predict_requires_formulation_path(self):
        from entrypoint import action_predict, build_parser

        args = build_parser().parse_args(
            [
                "--action",
                "predict",
                "--input",
                "/i",
                "--output",
                "/o",
                "--artifact-directory",
                "/i/artifacts/default",
            ]
        )
        with pytest.raises(ValueError, match="formulation_path is required"):
            action_predict(args)

    def test_predict_component_db_path_is_optional(self):
        from unittest.mock import patch

        from entrypoint import action_predict, build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--action",
                "predict",
                "--input",
                "/i",
                "--output",
                "/o",
                "--artifact-directory",
                "/i/artifacts/default",
                "--formulation-path",
                "/i/new_formulations.csv",
            ]
        )
        assert args.component_db_path is None
        with patch("entrypoint.predict_formulations") as mock_predict:
            mock_predict.return_value = {"n_predictions": 0, "output_files": {}}
            action_predict(args)
            mock_predict.assert_called_once()
            assert mock_predict.call_args.kwargs["component_db_path"] is None

    def test_predict_accepts_component_db_path(self):
        from entrypoint import build_parser

        args = build_parser().parse_args(
            [
                "--action",
                "predict",
                "--input",
                "/i",
                "--output",
                "/o",
                "--artifact-directory",
                "/i/artifacts/default",
                "--formulation-path",
                "/i/new_formulations.csv",
                "--component-db-path",
                "/i/chemical_db.csv",
            ]
        )
        assert args.component_db_path == "/i/chemical_db.csv"
