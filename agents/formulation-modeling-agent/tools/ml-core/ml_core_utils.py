#!/usr/bin/env python3
"""NobleAI ML Core utilities for Microsoft Discovery formulation modeling workflows.

Wraps noble-ml-core for training property prediction models on formulation data
(IDs + Proportions columns) inside Discovery container tools.
"""

from __future__ import annotations

import ast
import glob
import json
import logging
import os
import shutil
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from noble_ml_core import GLOBAL_FEATURIZER_REGISTRY, GLOBAL_MODEL_REGISTRY, GLOBAL_PROCESSOR_REGISTRY
from noble_ml_core.training.run_experiment import run_experiment
from noble_ml_core.artifact_tools import ArtifactLoggerType, LocalArtifactLogger
from noble_ml_core.visualization import get_all_plots
from noble_ml_core.registry import SPLIT_METHOD_REGISTRY
from noble_ml_core.data.formulation_dataloader import (
    LocalFormulationDataLoaderHyperparameterSet,
    _align_chemical_database,
    _align_formulation_dataframe,
)
from noble_ml_core.data.vip_formulation import (
    VipFormulationDataLoader,
    VipFormulationDataLoaderHyperparameterSet,
    is_vip_wide_formulation,
    load_vip_formulation_table,
    vip_to_lists,
)
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# User-uploaded formulation + component CSVs are mounted here in Discovery.
DATA_MOUNT = "/inputs"
INPUT_DIR = "/inputs"
OUTPUT_DIR = "/output"
WORK_DIR = "/workdir"

_HOST_PATH_PREFIXES = ("/Users/", "/home/")

# Column defaults match noble-ml-core formulation dataloaders (not Discovery-only).
_FORMULATION_DEFAULTS = LocalFormulationDataLoaderHyperparameterSet()
DEFAULT_FORMULATION_COLUMN = _FORMULATION_DEFAULTS.id_column
DEFAULT_QUANTITY_COLUMN = _FORMULATION_DEFAULTS.proportions_column
# SMILES is the chemical-featurizer convention (not a dataloader field).
DEFAULT_SMILES_COLUMN = "SMILES"
_DEFAULT_COMPONENT_DROP_COLUMNS = ("component_name", "Component UUID", "Component Name")
IGNORE_COMPONENTS = [
    "InvertibleProcessorComponent",
    "TorchModel",
    "FeaturizerComponent",
]

_REGISTRIES = {
    "featurizer": GLOBAL_FEATURIZER_REGISTRY,
    "model": GLOBAL_MODEL_REGISTRY,
    "processor": GLOBAL_PROCESSOR_REGISTRY,
}

def _resolve_registry(registry_name: str):
    """Resolves the registry: featurizer, model, processor."""
    if registry_name not in _REGISTRIES:
        raise ValueError(f"Unknown registry {registry_name!r}; expected one of {sorted(_REGISTRIES)}.")
    return _REGISTRIES[registry_name]

def list_components_and_descriptions(registry_name: str):
    """Lists the components and their descriptions in the registry: featurizer, model, processor.
    """
    registry = _resolve_registry(registry_name)
    response = {}
    for comp in registry.list_components():
        if comp in IGNORE_COMPONENTS:
            continue
        docstring = registry.get_component_metadata(comp).get("docstring") or ""
        response[comp] = docstring.replace("\n", "")
    return response

def list_postprocessors() -> Dict[str, str]:
    """List registered invertible processors."""
    response = {}
    for comp in GLOBAL_PROCESSOR_REGISTRY.list_components():
        if comp in IGNORE_COMPONENTS:
            continue
        metadata = GLOBAL_PROCESSOR_REGISTRY.get_component_metadata(comp) or {}
        if metadata.get("invertable"):
            docstring = metadata.get("docstring") or ""
            response[comp] = docstring.replace("\n", "")
    return response



def list_preprocessors() -> Dict[str, str]:
    """List processors, including chemical featurizers."""
    response = list_components_and_descriptions("processor")
    return response

def list_models(problem_type: str = "regression", multioutput: bool = False, null_targets: bool = False):
    """Lists the compatible models for a given problem type (regression, binary-classification, multiclass-classification), multioutput, and null targets.
    """
    response = {}
    for comp in GLOBAL_MODEL_REGISTRY.list_components():
        if comp in IGNORE_COMPONENTS:
            continue
        metadata = GLOBAL_MODEL_REGISTRY.get_component_metadata(comp)
        if problem_type not in metadata.get('supported_problem_types', []):
            continue
        if multioutput and not metadata.get("supports_multioutput"):
            continue
        if null_targets and not metadata.get("supports_null_targets"):
            continue
        response[comp] = (metadata.get("docstring") or "").replace("\n", "")
    return response

def list_split_methods() -> Dict[str, Dict[str, Any]]:
    """List registered dataloader split methods with signatures and context."""
    response: Dict[str, Dict[str, Any]] = {}
    for name in SPLIT_METHOD_REGISTRY.list_callables():
        metadata = SPLIT_METHOD_REGISTRY.get_callable_metadata(name) or {}
        response[name] = {
            "docstring": (metadata.get("docstring") or "").strip(),
            "signature": metadata.get("signature", ""),
            "parameters": list(metadata.get("parameters") or []),
        }
    return response


def list_split_method_config_parameters(split_method: str) -> List[str]:
    """List dataloader config fields consumed by a registered split method."""
    if not SPLIT_METHOD_REGISTRY.is_registered(split_method):
        raise ValueError(
            f"Unknown split method {split_method!r}. Available methods: "
            f"{SPLIT_METHOD_REGISTRY.list_callables()}"
        )
    metadata = SPLIT_METHOD_REGISTRY.get_callable_metadata(split_method) or {}
    dataloader_fields = {
        field_obj.name for field_obj in fields(VipFormulationDataLoaderHyperparameterSet)
    }
    return [
        name
        for name in (metadata.get("parameters") or [])
        if name in dataloader_fields
    ]
    

def list_available_models(
    problem_type: str = "regression",
    multioutput: bool = False,
    null_targets: bool = False,
) -> List[str]:
    """Installed model names compatible with the given problem type."""
    return list(list_models(problem_type, multioutput, null_targets).keys())


def _component_field_schema(registry, component_name: str) -> Tuple[type, Dict[str, Any]]:
    """Return a registered component's config class and field schema."""
    if not registry.is_registered(component_name):
        raise ValueError(f"Component {component_name!r} is not registered in {registry}.")

    metadata = registry.get_component_metadata(component_name) or {}
    config_class = metadata.get("config_class")
    if config_class is None:
        raise ValueError(f"Component {component_name!r} has no config class.")
    schema: Dict[str, Any] = {}
    for field_obj in fields(config_class):
        spec: Dict[str, Any] = {"type": field_obj.type}
        if field_obj.default is not MISSING:
            spec["default"] = field_obj.default
        elif field_obj.default_factory is not MISSING:
            spec["default"] = field_obj.default_factory()
        schema[field_obj.name] = spec
    return config_class, schema


def get_component_config(
    component_name: str, component_type: str
) -> Dict[str, Dict[str, Any]]:
    """Return serializable config field types/defaults for a model or processor."""
    registry = _resolve_registry(component_type)
    _config_class, schema = _component_field_schema(registry, component_name)
    result: Dict[str, Dict[str, Any]] = {}
    for name, spec in schema.items():
        field_type = spec["type"]
        result[name] = {"type": getattr(field_type, "__name__", str(field_type))}
        if "default" in spec:
            result[name]["default"] = spec["default"]
    return result


def get_component_hydra_target(
    component_name: str, component_type: str
) -> Dict[str, Any]:
    """Return the minimal Hydra block for a registered model or processor."""
    registry = _resolve_registry(component_type)
    if not registry.is_registered(component_name):
        raise ValueError(
            f"Component {component_name!r} is not registered as {component_type}."
        )
    meta = registry.get_component_metadata(component_name) or {}
    cls = meta["class"]
    config_class = meta["config_class"]
    return {
        "_target_": f"{cls.__module__}.{cls.__name__}",
        "config": {
            "_target_": f"{config_class.__module__}.{config_class.__name__}",
        },
    }


def get_dataclass_defaults(config_class: type) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    if hasattr(config_class, "__dataclass_fields__"):
        for field_obj in fields(config_class):
            if field_obj.default is not MISSING:
                defaults[field_obj.name] = field_obj.default
            elif field_obj.default_factory is not MISSING:
                defaults[field_obj.name] = field_obj.default_factory()
    return defaults


def build_formulation_data_config(
    formulation_file: str,
    target_columns: List[str],
    *,
    component_file_path: Optional[str] = None,
    id_column: str = DEFAULT_FORMULATION_COLUMN,
    proportions_column: str = DEFAULT_QUANTITY_COLUMN,
    input_columns: Optional[List[str]] = None,
    problem_type: str = "regression",
    split_method: str = "train_test_val",
    test_size: float = 0.2,
    random_state: int = 42,
    loader_type: str = "local",
    use_basenames: bool = False,
    **config_overrides: Any,
) -> Dict[str, Any]:
    """Build the standard noble-ml-core formulation ``data`` config block.

    Args:
        formulation_file: Formulation table path (mapped to ``file_path``).
            Prefer the mounted Discovery path, e.g. ``/inputs/<file>.csv``.
        target_columns: Columns to model.
        component_file_path: Optional component table path (``/inputs/<file>.csv``).
        id_column: Column containing component identifier lists.
        proportions_column: Column containing component proportion lists.
        input_columns: Model inputs. Defaults to ID and proportion columns.
            Include categorical covariates; they are one-hot encoded later.
        problem_type: Noble-ml-core problem type.
        split_method: Data splitting strategy.
        test_size: Holdout fraction.
        random_state: Split seed.
        loader_type: ``"local"`` or ``"vip"``. VIP additionally supports wide
            formulation CSVs and VIP component-table normalization.
        use_basenames: If True, store only path basenames. Default False so
            authored YAML keeps ``/inputs/<file>.csv`` after remounting.
        **config_overrides: Additional fields accepted by the selected
            dataloader hyperparameter set (for example ``num_folds`` or
            ``vip_id_column``).
    """
    if not formulation_file or not str(formulation_file).strip():
        raise ValueError("formulation_file is required.")
    if not target_columns:
        raise ValueError("target_columns must contain at least one column.")

    loader_types = {
        "local": (
            "noble_ml_core.data.formulation_dataloader.LocalFormulationDataLoader",
            LocalFormulationDataLoaderHyperparameterSet,
        ),
        "vip": (
            "noble_ml_core.data.vip_formulation.VipFormulationDataLoader",
            VipFormulationDataLoaderHyperparameterSet,
        ),
    }
    normalized_loader_type = str(loader_type).strip().lower()
    if normalized_loader_type not in loader_types:
        raise ValueError(
            f"Unknown loader_type {loader_type!r}; expected one of "
            f"{sorted(loader_types)}."
        )
    if not SPLIT_METHOD_REGISTRY.is_registered(split_method):
        raise ValueError(
            f"Unknown split_method {split_method!r}; expected one of "
            f"{SPLIT_METHOD_REGISTRY.list_callables()}."
        )

    dataloader_target, config_class = loader_types[normalized_loader_type]
    resolved_inputs = list(input_columns or [id_column, proportions_column])
    missing_formulation_columns = [
        column
        for column in (id_column, proportions_column)
        if column not in resolved_inputs
    ]
    if missing_formulation_columns:
        raise ValueError(
            "input_columns must include the formulation ID and proportion "
            f"columns: {missing_formulation_columns}"
        )

    def config_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        value = str(path)
        return os.path.basename(value) if use_basenames else value

    config_values: Dict[str, Any] = {
        "file_path": config_path(formulation_file),
        "component_file_path": config_path(component_file_path),
        "id_column": id_column,
        "proportions_column": proportions_column,
        "input_columns": resolved_inputs,
        "target_columns": list(target_columns),
        "problem_type": problem_type,
        "split_method": split_method,
        "test_size": test_size,
        "random_state": random_state,
        **config_overrides,
    }
    valid_fields = {field_obj.name for field_obj in fields(config_class)}
    unknown_fields = sorted(set(config_values) - valid_fields)
    if unknown_fields:
        raise ValueError(
            f"Unsupported {normalized_loader_type} dataloader config fields: "
            f"{unknown_fields}"
        )

    # Invoke the noble-ml-core config class so invalid values fail while the
    # config is authored rather than during the execute action.
    config_class(**config_values)
    return {
        "dataloader": {"_target_": dataloader_target},
        "config": {
            "_target_": f"{config_class.__module__}.{config_class.__name__}",
            **config_values,
        },
    }


_REQUIRED_EXPERIMENT_KEYS = (
    "data",
    "preprocessing_pipeline",
    "postprocessing_pipeline",
    "model",    
)

_DATA_PATH_KEYS = ("file_path", "component_file_path")


def _is_host_path(path: str) -> bool:
    """Return True for laptop/host paths that will not exist in the container."""
    text = str(path).strip().replace("\\", "/")
    if text.startswith("~"):
        return True
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return True
    return any(text.startswith(prefix) for prefix in _HOST_PATH_PREFIXES)


def _config_has_formulation_processor(cfg: Any) -> bool:
    processors = OmegaConf.select(cfg, "preprocessing_pipeline.processors")
    if not processors:
        return False
    return "FormulationProcessor" in OmegaConf.to_yaml(processors)


def create_config_from_yaml_string(yaml_string: str) -> DictConfig:
    return OmegaConf.create(yaml_string)


def save_experiment_config(cfg: Any, path: str) -> str:
    """Persist a Hydra/OmegaConf experiment or sweep config to YAML."""
    OmegaConf.save(cfg, path)
    logging.info("Saved experiment config to %s", path)
    return path


def validate_experiment_config(
    cfg: Any,
    *,
    require_tune: bool = False,
) -> Dict[str, Any]:
    """Check that a config has the noble-ml-core experiment sections.

    Sweep configs must also include a ``tune`` block when ``require_tune=True``.
    """
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg) if os.path.isfile(cfg) else OmegaConf.create(cfg)
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    errors: List[str] = []
    warnings: List[str] = []
    missing = [key for key in _REQUIRED_EXPERIMENT_KEYS if key not in cfg]
    if missing:
        errors.append(f"Missing required keys: {missing}")
    has_tune = "tune" in cfg
    if require_tune and not has_tune:
        errors.append(
            "Sweep configs must include a 'tune' block. "
            "Do not call run_hyperparameter_sweep until the YAML defines search spaces."
        )
    data_config = OmegaConf.select(cfg, "data.config")
    if data_config is None:
        errors.append("data.config is required")
    else:
        if OmegaConf.select(data_config, "file_path") in (None, ""):
            errors.append("data.config.file_path is required")
        for key in _DATA_PATH_KEYS:
            value = OmegaConf.select(data_config, key)
            if not value:
                continue
            path_str = str(value)
            if _is_host_path(path_str):
                errors.append(
                    f"data.config.{key} {path_str!r} looks like a host path. "
                    f"Use the mounted {DATA_MOUNT}/<file>.csv path."
                )
            elif os.path.basename(path_str) == path_str:
                warnings.append(
                    f"data.config.{key} {path_str!r} is a basename; prefer "
                    f"{DATA_MOUNT}/{path_str}."
                )
        target_columns = OmegaConf.select(data_config, "target_columns")
        if not target_columns:
            errors.append("data.config.target_columns must not be empty")
        split_method = OmegaConf.select(data_config, "split_method")
        if split_method and not SPLIT_METHOD_REGISTRY.is_registered(str(split_method)):
            errors.append(
                f"Unknown data.config.split_method {split_method!r}; expected one of "
                f"{SPLIT_METHOD_REGISTRY.list_callables()}"
            )

    if "preprocessing_pipeline" in cfg and not _config_has_formulation_processor(cfg):
        errors.append(
            "preprocessing_pipeline must include FormulationProcessor "
            "(with a nested component_pipeline)."
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "has_tune": has_tune,
    }


def load_component_database(
    csv_path: str,
    id_column: str = DEFAULT_FORMULATION_COLUMN,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
) -> pd.DataFrame:
    """Load optional component metadata (IDs + SMILES) for chemical featurizers."""
    csv_path = _resolve_input_path(csv_path)
    df = _load_component_db_via_dataloader(csv_path, id_column=id_column)
    if smiles_column not in df.columns:
        raise ValueError(
            f"Component database must include a '{smiles_column}' column for chemical featurizers. "
            f"Available columns: {list(df.columns)}"
        )
    duplicates = int(df.duplicated(subset=[id_column]).sum())
    if duplicates:
        raise ValueError(
            f"Component database has {duplicates} duplicate entries in '{id_column}'"
        )
    logging.info(
        "Loaded component database with %s components from %s",
        len(df),
        csv_path,
    )
    return df


def _load_component_db_via_dataloader(
    csv_path: str,
    *,
    id_column: str = DEFAULT_FORMULATION_COLUMN,
    vip_id_column: str = "id",
    drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Align a component CSV using ``VipFormulationDataLoader.load_component_db``."""
    drop_cols = (
        list(drop_columns)
        if drop_columns is not None
        else list(_DEFAULT_COMPONENT_DROP_COLUMNS)
    )
    config = VipFormulationDataLoaderHyperparameterSet(
        component_file_path=csv_path,
        id_column=id_column,
        vip_id_column=vip_id_column,
        component_db_drop_columns=drop_cols,
    )
    db = VipFormulationDataLoader.load_component_db(config)
    if db is None:
        raise ValueError(f"Failed to load component database from {csv_path}")
    if id_column not in db.columns:
        # Vip loader only auto-renames ``vip_id_column``; handle Discovery aliases.
        return align_component_database(
            pd.read_csv(csv_path),
            id_column=id_column,
            drop_columns=drop_cols,
            vip_id_column=vip_id_column,
        )
    return db

def build_component_config(component_name: str, component_type: str, component_config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a validated Hydra block for a registered model or processor."""
    registry = _resolve_registry(component_type)
    if not registry.is_registered(component_name):
        raise ValueError(
            f"Component {component_name} is not a valid option for registry {component_type}."
        )
    meta = registry.get_component_metadata(component_name) or {}
    cls = meta["class"]
    config_class, _config_schema = _component_field_schema(registry, component_name)
    defaults = get_dataclass_defaults(config_class)
    valid_field_names = {field_obj.name for field_obj in fields(config_class)}
    overrides = component_config_overrides or {}
    valid_config = {key: value for key, value in {**defaults, **overrides}.items() if key in valid_field_names}
    config_class(**valid_config)
    return {
        "_target_": f"{cls.__module__}.{cls.__name__}",
        "config": {
            "_target_": f"{config_class.__module__}.{config_class.__name__}",
            **valid_config,
        },
    }

def quick_setup(input_dir="/inputs", output_dir="/output", work_dir="/workdir"):
    """Initialize logging, create directories, and copy input files."""
    global INPUT_DIR, OUTPUT_DIR, WORK_DIR
    INPUT_DIR = os.path.abspath(input_dir)
    OUTPUT_DIR = os.path.abspath(output_dir)
    WORK_DIR = os.path.abspath(work_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for directory in (WORK_DIR, OUTPUT_DIR):
        os.makedirs(directory, exist_ok=True)
    os.chdir(WORK_DIR)
    _copy_input_files()
    logging.info("Working directory: %s", WORK_DIR)
    logging.info(
        "Input files: %s",
        os.listdir(INPUT_DIR) if os.path.exists(INPUT_DIR) else "(none)",
    )


def _copy_input_files():
    if os.path.realpath(INPUT_DIR) == os.path.realpath(WORK_DIR):
        return
    if not os.path.exists(INPUT_DIR):
        return
    for filepath in glob.glob(os.path.join(INPUT_DIR, "*")):
        if os.path.isfile(filepath):
            shutil.copy(filepath, WORK_DIR)


def quick_finish():
    """Copy key output files from workdir to the output directory."""
    if os.path.realpath(WORK_DIR) == os.path.realpath(OUTPUT_DIR):
        return
    patterns = [
        "*.json",
        "*.csv",
        "*.png",
        "*.svg",
        "*.pkl",
        "*.pickle",
        "*.html",
        "*.yaml",
        "*.log",
        "*.out",
    ]
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(WORK_DIR, pattern)):
            shutil.copy(filepath, OUTPUT_DIR)
    logging.info("Outputs copied to %s", OUTPUT_DIR)


def save_final_results(
    results: Dict,
    output_files: Optional[Dict] = None,
    file_descriptions: Optional[Dict] = None,
    status: str = "completed",
):
    """Save structured results to final_results.json (required for every script)."""
    payload = {"status": status, "summary": _json_safe(results)}
    if output_files:
        payload["output_files"] = _json_safe(output_files)
    if file_descriptions:
        payload["file_descriptions"] = _json_safe(file_descriptions)

    out_path = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logging.info("Saved final_results.json to %s", out_path)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _parse_list_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Handle CSV-escaped JSON strings like "\"[1, 2]\"" or '"[1, 2]"'
        for _ in range(2):
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
                text = text[1:-1].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return parsed
    return value


def _resolve_input_path(path: str) -> str:
    """Resolve a data file path after quick_setup may have changed cwd."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    basename = os.path.basename(path)
    for candidate in (
        os.path.join(DATA_MOUNT, basename),
        os.path.join(WORK_DIR, basename),
        os.path.join(INPUT_DIR, basename),
        os.path.join(DATA_MOUNT, path),
        os.path.join(WORK_DIR, path),
        os.path.join(INPUT_DIR, path),
    ):
        if os.path.exists(candidate):
            return candidate
    return path


def _resolve_data_path(path: str, input_directory: Optional[str] = None) -> str:
    """Resolve an explicit data file path. Does not guess filenames."""
    if not path or not str(path).strip():
        raise ValueError("A file path is required.")
    path = str(path).strip()
    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    basename = os.path.basename(path)
    candidates.append(os.path.join(DATA_MOUNT, basename))
    if input_directory:
        candidates.append(os.path.join(input_directory, path))
        candidates.append(os.path.join(input_directory, basename))
    candidates.append(_resolve_input_path(path))
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"File not found: {path}. Pass an explicit path (absolute or relative "
        f"to {DATA_MOUNT} or the input directory)."
    )


def _rebase_path(path_str: str, input_dir: str) -> str:
    """Resolve a config data path, preferring the mounted ``/inputs`` CSVs."""
    if not path_str:
        return path_str
    basename = os.path.basename(str(path_str))
    candidates = [
        os.path.join(DATA_MOUNT, basename),
        os.path.join(input_dir, basename),
        os.path.join(input_dir, str(path_str)),
        os.path.join(INPUT_DIR, basename),
        os.path.join(WORK_DIR, basename),
        str(path_str),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Could not resolve data file {path_str!r} after remounting. "
        f"Searched {DATA_MOUNT!r}, input directory {input_dir!r}, "
        f"INPUT_DIR {INPUT_DIR!r}, and WORK_DIR {WORK_DIR!r}."
    )


def _rebase_config_paths(
    cfg: DictConfig,
    input_dir: str,
    output_dir: str,
) -> DictConfig:
    """Rewrite data paths for a new container and point artifact logging at output_dir."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    data_config = OmegaConf.select(cfg, "data.config")
    if data_config is not None:
        for key in _DATA_PATH_KEYS:
            value = OmegaConf.select(data_config, key)
            if value:
                data_config[key] = _rebase_path(str(value), input_dir)

    artifact_base = os.path.abspath(os.path.join(output_dir, "model_artifacts"))
    existing = OmegaConf.select(cfg, "artifact_logging")
    if existing is None:
        cfg.artifact_logging = OmegaConf.create(
            {
                "artifact_logger_type": "${artifactlogger:LOCAL}",
                "config": {"base_path": artifact_base, "overwrite": True},
            }
        )
    else:
        if OmegaConf.select(cfg, "artifact_logging.config") is None:
            cfg.artifact_logging.config = OmegaConf.create({})
        cfg.artifact_logging.config.base_path = artifact_base
    return cfg


def _load_and_rebase_config(
    config_path: str,
    input_dir: str,
    output_dir: str,
) -> Tuple[DictConfig, str]:
    resolved = _resolve_input_path(config_path)
    if not os.path.isfile(resolved):
        alt = os.path.join(input_dir, os.path.basename(config_path))
        if os.path.isfile(alt):
            resolved = alt
        else:
            raise FileNotFoundError(
                f"config_path not found: {config_path}. "
                "Mount the coding-environment output (containing the YAML) as input_directory."
            )
    cfg = OmegaConf.load(resolved)
    return _rebase_config_paths(cfg, input_dir, output_dir), resolved


# ---------------------------------------------------------------------------
# Data-format detection, VIP conversion, component DB alignment
# Delegates to noble-ml-core VipFormulationDataLoader / vip_to_lists helpers.
# ---------------------------------------------------------------------------



def detect_data_format(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect list-format vs VIP platform-wide formulation columns."""
    has_list = (
        DEFAULT_FORMULATION_COLUMN in df.columns
        and DEFAULT_QUANTITY_COLUMN in df.columns
    )
    has_platform = is_vip_wide_formulation(df)
    if has_list and has_platform:
        fmt = "ambiguous"
    elif has_list:
        fmt = "list"
    elif has_platform:
        fmt = "platform_wide"
    else:
        fmt = "unknown"
    return {
        "format": fmt,
        "has_list_columns": has_list,
        "has_platform_columns": has_platform,
        "columns": list(df.columns),
    }


def convert_platform_to_lists(
    df: pd.DataFrame,
    normalize_proportions: bool = False,
) -> pd.DataFrame:
    """Convert VIP wide columns to IDs / Proportions lists."""
    converted = vip_to_lists(df)
    if normalize_proportions and DEFAULT_QUANTITY_COLUMN in converted.columns:
        def _normalize(props: Any) -> Any:
            if not isinstance(props, list) or not props:
                return props
            total = float(sum(props))
            if total <= 0:
                return props
            return [float(p) / total for p in props]

        converted = converted.copy()
        converted[DEFAULT_QUANTITY_COLUMN] = converted[DEFAULT_QUANTITY_COLUMN].apply(
            _normalize
        )
    return converted


def align_component_database(
    df: pd.DataFrame,
    id_column: str = DEFAULT_FORMULATION_COLUMN,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
    drop_columns: Optional[List[str]] = None,
    vip_id_column: str = "id",
) -> pd.DataFrame:
    """Normalize a component database the same way ``VipFormulationDataLoader`` does."""
    out = df.copy()
    if id_column not in out.columns:
        for candidate in (
            vip_id_column,
            "IDs",
            "ID",
            "id",
            "component_id",
            "Component ID",
        ):
            if candidate in out.columns:
                out = out.rename(columns={candidate: id_column})
                break
    if id_column not in out.columns:
        raise ValueError(
            f"Component database missing id column '{id_column}'. "
            f"Available: {list(out.columns)}"
        )

    drop_cols = (
        list(drop_columns)
        if drop_columns is not None
        else list(_DEFAULT_COMPONENT_DROP_COLUMNS)
    )
    present_drops = [c for c in drop_cols if c in out.columns and c != id_column]
    if present_drops:
        out = out.drop(columns=present_drops)

    out = _align_chemical_database(out, id_column=id_column)

    duplicates = int(out.duplicated(subset=[id_column]).sum())
    if duplicates:
        raise ValueError(
            f"Component database has {duplicates} duplicate entries in '{id_column}'"
        )

    if smiles_column in out.columns:
        missing_smiles = int(out[smiles_column].isna().sum())
        if missing_smiles:
            logging.warning(
                "Component database has %s missing SMILES values", missing_smiles
            )
    return out


def _is_categorical_series(series: pd.Series) -> bool:
    return bool(
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def assess_model_readiness(
    df: pd.DataFrame,
    component_db: Optional[pd.DataFrame] = None,
    candidate_targets: Optional[List[str]] = None,
    formulation_column: str = DEFAULT_FORMULATION_COLUMN,
    quantity_column: str = DEFAULT_QUANTITY_COLUMN,
    **kwargs,
) -> Dict[str, Any]:
    """Assess whether formulation data is ready for modeling."""
    excluded = {formulation_column, quantity_column}
    numeric_cols = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    targets_inferred = candidate_targets is None
    if candidate_targets is None:
        candidate_targets = list(numeric_cols)
    else:
        candidate_targets = [c for c in candidate_targets if c in df.columns]

    covariates = [c for c in numeric_cols if c not in candidate_targets]
    categorical_covariates = [
        col
        for col in df.columns
        if col not in excluded
        and col not in candidate_targets
        and _is_categorical_series(df[col])
    ]
    profile = profile_dataset(df, target_columns=candidate_targets or None)

    warnings: List[str] = []
    blockers: List[str] = []
    next_actions: List[str] = []

    n_rows = int(len(df))
    if n_rows < 10:
        blockers.append(f"Only {n_rows} rows; need at least 10 for a meaningful split.")
    elif n_rows < 50:
        warnings.append(f"Small dataset ({n_rows} rows); hold-out metrics may be noisy.")

    if not candidate_targets:
        blockers.append("No numeric target candidates found.")
    else:
        for col in candidate_targets:
            missing = int(df[col].isna().sum())
            if missing:
                warnings.append(f"Target '{col}' has {missing} missing values.")
            if float(df[col].std()) == 0.0:
                blockers.append(f"Target '{col}' has zero variance.")

    # Duplicate formulation signatures (same IDs + proportions)
    try:
        sig = df.apply(
            lambda r: (
                tuple(r[formulation_column]),
                tuple(np.round(r[quantity_column], 6)),
            ),
            axis=1,
        )
        n_dup = int(sig.duplicated().sum())
        if n_dup:
            warnings.append(f"{n_dup} duplicate formulation rows detected.")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not check duplicate formulations: {exc}")

    has_smiles = bool(
        component_db is not None and DEFAULT_SMILES_COLUMN in component_db.columns
    )

    next_actions.append(
        "Choose model and processors from list_models(), list_preprocessors(), "
        "and get_component_config() given this data profile."
    )
    if targets_inferred and len(candidate_targets) > 1:
        next_actions.append(
            "Multiple numeric columns found; pass the user-specified target_columns "
            "rather than treating every numeric column as a target."
        )
    if categorical_covariates:
        next_actions.append(
            "One-hot encode categorical columns with OneHotEncoderProcessor: "
            f"{categorical_covariates}."
        )
        next_actions.append(
            "Drop unused columns with FeatureSelectorProcessor "
            "(dropped_columns or selected_columns)."
        )
    elif covariates or has_smiles:
        next_actions.append(
            "Drop unused columns with FeatureSelectorProcessor "
            "(dropped_columns or selected_columns)."
        )

    if blockers:
        verdict = "not_ready"
    elif warnings:
        verdict = "conditionally_ready"
    else:
        verdict = "ready"

    return {
        "verdict": verdict,
        "target_candidates": candidate_targets,
        "covariates": covariates,
        "categorical_covariates": categorical_covariates,
        "warnings": warnings,
        "blockers": blockers,
        "next_actions": next_actions,
        "dataset_profile": profile,
        "n_rows": n_rows,
        "has_component_smiles": has_smiles,
    }


def prepare_formulation_inputs(
    formulation_path: str,
    component_db_path: Optional[str] = None,
    *,
    output_dir: Optional[str] = None,
    target_columns: Optional[List[str]] = None,
    normalize_proportions: bool = False,
) -> Dict[str, Any]:
    """Orchestrate ingest: detect format, convert if needed, align component DB."""
    formulation_path = _resolve_input_path(formulation_path)
    # Preserve VIP identifier string dtypes (avoids float coercion of IDs).
    raw_df = load_vip_formulation_table(formulation_path)
    format_info = detect_data_format(raw_df)
    output_dir = output_dir or WORK_DIR
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "format_detection": format_info,
        "source_formulation_path": formulation_path,
        "formulation_path": formulation_path,
        "component_db_path": component_db_path,
        "converted": False,
        "training_pickle_path": None,
        "aligned_component_db_path": None,
        "list_format_csv_path": None,
        "readiness": None,
        "warnings": [],
        "output_files": {},
    }

    if format_info["format"] == "ambiguous":
        result["warnings"].append(
            "Both list and platform-wide columns detected; preferring list columns."
        )
    if format_info["format"] == "unknown":
        raise ValueError(
            "Could not detect formulation format; expected IDs/Proportions or "
            "component_N_identifier/component_N_amount columns. "
            f"Available: {format_info['columns']}"
        )

    if format_info["format"] == "platform_wide":
        list_df = convert_platform_to_lists(
            raw_df, normalize_proportions=normalize_proportions
        )
        list_csv = out / "formulation_list_format.csv"
        # Persist list columns as JSON strings for CSV round-trip
        persist_df = list_df.copy()
        persist_df[DEFAULT_FORMULATION_COLUMN] = persist_df[
            DEFAULT_FORMULATION_COLUMN
        ].apply(json.dumps)
        persist_df[DEFAULT_QUANTITY_COLUMN] = persist_df[DEFAULT_QUANTITY_COLUMN].apply(
            json.dumps
        )
        persist_df.to_csv(list_csv, index=False)
        result["converted"] = True
        result["list_format_csv_path"] = str(list_csv)
        result["formulation_path"] = str(list_csv)
        result["output_files"]["list_format_csv"] = str(list_csv)
        working_path = str(list_csv)
    else:
        working_path = formulation_path

    component_db = None
    if component_db_path:
        component_db_path = _resolve_input_path(component_db_path)
        component_db = _load_component_db_via_dataloader(component_db_path)
        aligned_path = out / "ml_core_component_database.csv"
        component_db.to_csv(aligned_path, index=False)
        result["component_db_path"] = str(aligned_path)
        result["aligned_component_db_path"] = str(aligned_path)
        result["output_files"]["aligned_component_db"] = str(aligned_path)

    df = load_formulation_csv(working_path, target_columns=target_columns)
    readiness = assess_model_readiness(
        df,
        component_db=component_db,
        candidate_targets=target_columns,
    )
    result["readiness"] = readiness

    resolved_targets = target_columns or readiness.get("target_candidates") or None
    pickle_path, input_cols, resolved_targets = prepare_training_file(
        df,
        output_path=str(out / "formulation_training_ready.csv"),
        target_columns=resolved_targets,
    )
    result["training_pickle_path"] = pickle_path
    result["input_columns"] = input_cols
    result["target_columns"] = resolved_targets
    result["output_files"]["training_pickle"] = pickle_path
    result["warnings"].extend(readiness.get("warnings", []))
    return result


def prepare_formulation_data(
    input_directory: str,
    output_directory: str,
    *,
    formulation_path: str,
    component_db_path: Optional[str] = None,
    target_columns: Optional[List[str]] = None,
    normalize_proportions: bool = False,
) -> Dict[str, Any]:
    """Prepare formulation (and optional component) files from explicit paths.

    ``formulation_path`` is required. ``component_db_path`` is optional; when
    omitted, no component table is loaded. Paths may be absolute or relative to
    ``input_directory``. Filenames are never inferred.
    """
    if not formulation_path:
        raise ValueError("formulation_path is required.")
    input_directory = _resolve_input_path(input_directory)
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    formulation_resolved = _resolve_data_path(formulation_path, input_directory)
    component_resolved = (
        _resolve_data_path(component_db_path, input_directory)
        if component_db_path
        else None
    )

    if isinstance(target_columns, str):
        target_columns = [c.strip() for c in target_columns.split(",") if c.strip()]

    result = prepare_formulation_inputs(
        formulation_resolved,
        component_db_path=component_resolved,
        output_dir=output_directory,
        target_columns=target_columns,
        normalize_proportions=normalize_proportions,
    )
    result["input_directory"] = input_directory
    result["output_directory"] = output_directory
    return result


def load_formulation_csv(
    csv_path: str,
    target_columns: Optional[List[str]] = None,
    formulation_column: str = DEFAULT_FORMULATION_COLUMN,
    quantity_column: str = DEFAULT_QUANTITY_COLUMN,
    max_rows: Optional[int] = None,
    require_targets: bool = True,
) -> pd.DataFrame:
    """Load and validate a formulation CSV dataset."""
    csv_path = _resolve_input_path(csv_path)
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if formulation_column not in df.columns:
        raise ValueError(
            f"Formulation column '{formulation_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if quantity_column not in df.columns:
        raise ValueError(
            f"Quantity column '{quantity_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df[formulation_column] = df[formulation_column].apply(_parse_list_value)
    df[quantity_column] = df[quantity_column].apply(_parse_list_value)
    df = _align_formulation_dataframe(
        df,
        id_column=formulation_column,
        proportions_column=quantity_column,
    )

    invalid = df[
        df[formulation_column].isna()
        | df[quantity_column].isna()
        | df[formulation_column].apply(lambda x: not isinstance(x, list) or len(x) == 0)
        | df[quantity_column].apply(lambda x: not isinstance(x, list) or len(x) == 0)
    ]
    if not invalid.empty:
        logging.warning("Dropping %s rows with invalid formulation lists", len(invalid))
        df = df.drop(index=invalid.index)

    length_mismatch = df[
        df[formulation_column].apply(len) != df[quantity_column].apply(len)
    ]
    if not length_mismatch.empty:
        raise ValueError(
            f"{len(length_mismatch)} rows have mismatched IDs and Proportions lengths"
        )

    if target_columns is None:
        excluded = {formulation_column, quantity_column}
        target_columns = [
            col
            for col in df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]
    if not target_columns:
        if require_targets:
            raise ValueError("No target columns found. Specify target_columns explicitly.")
        logging.info("Loaded %s formulation rows (no targets)", len(df))
        return df.reset_index(drop=True)

    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found: {missing_targets}")

    df = df.dropna(subset=target_columns, how="any").reset_index(drop=True)
    logging.info(
        "Loaded %s formulation rows with targets %s",
        len(df),
        target_columns,
    )
    return df


def validate_formulation_schema(
    df: pd.DataFrame,
    formulation_column: str = DEFAULT_FORMULATION_COLUMN,
    quantity_column: str = DEFAULT_QUANTITY_COLUMN,
) -> Dict[str, Any]:
    """Return a lightweight validation summary for formulation data."""
    if df.empty:
        return {
            "n_rows": 0,
            "n_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "n_unique_components": 0,
            "components_per_formulation": {"min": 0, "max": 0, "mean": 0.0},
            "formulation_column": formulation_column,
            "quantity_column": quantity_column,
        }

    n_components = df[formulation_column].apply(len)
    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "n_unique_components": int(
            len({comp for ids in df[formulation_column] for comp in ids})
        ),
        "components_per_formulation": {
            "min": int(n_components.min()),
            "max": int(n_components.max()),
            "mean": float(n_components.mean()),
        },
        "formulation_column": formulation_column,
        "quantity_column": quantity_column,
    }


def profile_dataset(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Summarize dataset shape, targets, and missing values."""
    if target_columns is None:
        target_columns = [
            col
            for col in df.columns
            if col not in (DEFAULT_FORMULATION_COLUMN, DEFAULT_QUANTITY_COLUMN)
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    profile = validate_formulation_schema(df)
    profile["target_columns"] = target_columns
    profile["target_stats"] = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "missing": int(df[col].isna().sum()),
        }
        for col in target_columns
        if col in df.columns
    }
    return profile


def prepare_training_file(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    target_columns: Optional[List[str]] = None,
    formulation_column: str = DEFAULT_FORMULATION_COLUMN,
    quantity_column: str = DEFAULT_QUANTITY_COLUMN,
) -> Tuple[str, List[str], List[str]]:
    """Write a training CSV, keeping numeric and categorical covariate columns."""
    if output_path is None:
        output_path = os.path.join(WORK_DIR, "formulation_training_ready.csv")

    if target_columns is None:
        target_columns = [
            col
            for col in df.columns
            if col not in (formulation_column, quantity_column)
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    extra_cols = [
        col
        for col in df.columns
        if col not in (formulation_column, quantity_column)
        and col not in target_columns
    ]
    input_columns = [formulation_column, quantity_column] + extra_cols

    training_df = df[input_columns + target_columns].copy()
    training_df.to_csv(output_path, index=False)
    logging.info("Wrote training CSV to %s", output_path)
    return output_path, input_columns, target_columns


def _extract_flat_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    selected = metrics
    if "ensemble" in metrics and isinstance(metrics["ensemble"], dict):
        selected = metrics["ensemble"]
    elif "average" in metrics and isinstance(metrics["average"], dict):
        selected = metrics["average"]
    elif "fold_0" in metrics and isinstance(metrics["fold_0"], dict):
        selected = metrics["fold_0"]
    flattened = dict(selected)
    if "score" in metrics:
        flattened["score"] = metrics["score"]
    return flattened


def _collect_logged_plots(output_directory: str) -> Dict[str, str]:
    plots_dir = Path(output_directory) / "model_artifacts" / "plots"
    if not plots_dir.is_dir():
        return {}
    return {path.stem: str(path) for path in sorted(plots_dir.iterdir()) if path.is_file()}


def _save_plots_from_artifacts(
    artifacts: Dict[str, Any],
    cfg: DictConfig,
    output_directory: str,
) -> Dict[str, str]:
    """Persist noble-ml-core plots. Prefer LocalArtifactLogger output; else get_all_plots."""
    existing = _collect_logged_plots(output_directory)
    if existing:
        return existing
    if not artifacts:
        return {}
    fold_key = "fold_0" if "fold_0" in artifacts else next(iter(artifacts))
    fold = artifacts[fold_key]
    problem_type = str(OmegaConf.select(cfg, "data.config.problem_type") or "regression")
    figs = get_all_plots(
        Y_actual=fold["Y"],
        Y_pred_for_metrics=fold["Y_pred"],
        problem_type=problem_type,
        plot_prefix=str(fold_key),
    )
    plots_dir = Path(output_directory) / "model_artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_files: Dict[str, str] = {}
    for name, fig in figs.items():
        if hasattr(fig, "write_html"):
            path = plots_dir / f"{name}.html"
            fig.write_html(str(path))
        elif hasattr(fig, "savefig"):
            path = plots_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        else:
            continue
        output_files[name] = str(path)
    return output_files


def _default_artifact_dir(output_directory: str) -> Optional[str]:
    default_artifact = Path(output_directory) / "model_artifacts" / "artifacts" / "default"
    if default_artifact.exists():
        return str(default_artifact)
    return None


def _apply_artifact_logging(cfg: DictConfig, output_directory: str, save_artifacts: bool) -> None:
    if save_artifacts:
        cfg.artifact_logging = OmegaConf.create(
            {
                "artifact_logger_type": "${artifactlogger:LOCAL}",
                "config": {
                    "base_path": os.path.join(output_directory, "model_artifacts"),
                    "overwrite": True,
                },
            }
        )
    else:
        cfg.artifact_logging = OmegaConf.create(
            {"artifact_logger_type": "NOOP", "use_artifact_logging": False}
        )


def _run_config_experiment(
    cfg: DictConfig,
    output_directory: str,
    *,
    save_artifacts: bool = True,
    generate_plots: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """Call noble-ml-core run_experiment and collect metrics/plots/artifacts."""
    from noble_ml_core.resolvers import register_common_resolvers

    register_common_resolvers()
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    _apply_artifact_logging(cfg, output_directory, save_artifacts)

    prev = os.getcwd()
    os.chdir(output_directory)
    try:
        artifacts, metrics = run_experiment(
            cfg,
            artifact_logger_type=(
                ArtifactLoggerType.LOCAL if save_artifacts else ArtifactLoggerType.NOOP
            ),
            generate_plots=generate_plots,
        )
    finally:
        os.chdir(prev)

    output_files: Dict[str, str] = {}
    if generate_plots:
        output_files.update(_save_plots_from_artifacts(artifacts, cfg, output_directory))
    artifact_dir = _default_artifact_dir(output_directory)
    if artifact_dir:
        output_files["model_artifacts"] = artifact_dir
    return artifacts, metrics, output_files


def run_experiment_action(
    input_directory: str,
    output_directory: str,
    *,
    config_path: str,
    save_artifacts: bool = True,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Run noble-ml-core training from a Hydra YAML authored in the coding environment."""
    if not config_path:
        raise ValueError(
            "config_path is required. Author experiment_config.yaml in the coding "
            "environment, then pass it to this action."
        )
    cfg, _resolved = _load_and_rebase_config(
        config_path, input_directory, output_directory
    )
    validation = validate_experiment_config(cfg, require_tune=False)
    if not validation["ok"]:
        raise ValueError("Invalid experiment config: " + "; ".join(validation["errors"]))

    _artifacts, metrics, output_files = _run_config_experiment(
        cfg,
        output_directory,
        save_artifacts=save_artifacts,
        generate_plots=generate_plots,
    )
    saved_config = os.path.join(output_directory, "experiment_config.yaml")
    save_experiment_config(cfg, saved_config)
    output_files["experiment_config"] = saved_config
    result: Dict[str, Any] = {
        "metrics": _extract_flat_metrics(metrics),
        "config_path": saved_config,
        "output_files": output_files,
    }
    if "model_artifacts" in output_files:
        result["artifact_dir"] = output_files["model_artifacts"]
    return result


def run_hyperparameter_sweep(
    input_directory: str,
    output_directory: str,
    *,
    config_path: str,
) -> Dict[str, Any]:
    """Run a Ray Tune sweep from a Hydra YAML that already includes a tune block."""
    from noble_ml_core.resolvers import register_common_resolvers
    from noble_ml_core.training.hyperparameter_tuning.run_sweep import run_sweep

    if not config_path:
        raise ValueError(
            "config_path is required. Author sweep_config.yaml (including a tune block) "
            "in the coding environment, then pass it to this action."
        )
    register_common_resolvers()
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    cfg, _resolved = _load_and_rebase_config(
        config_path, input_directory, output_directory
    )
    validation = validate_experiment_config(cfg, require_tune=True)
    if not validation["ok"]:
        raise ValueError("Invalid sweep config: " + "; ".join(validation["errors"]))

    sweep_dir = os.path.join(output_directory, "sweep")
    Path(sweep_dir).mkdir(parents=True, exist_ok=True)
    save_experiment_config(cfg, os.path.join(sweep_dir, "sweep_config.yaml"))
    results = run_sweep(cfg=cfg, output_dir=sweep_dir)
    best_path = os.path.join(sweep_dir, "best_config.yaml")
    payload: Dict[str, Any] = {
        "sweep_directory": sweep_dir,
        "best_config_path": best_path if os.path.isfile(best_path) else None,
        "output_files": {
            "sweep_config": os.path.join(sweep_dir, "sweep_config.yaml"),
        },
    }
    if payload["best_config_path"]:
        payload["output_files"]["best_config"] = payload["best_config_path"]
    payload["n_trials"] = len(results) if results is not None else None
    return payload


def train_best_from_sweep(
    sweep_directory: str,
    output_directory: str,
    *,
    input_directory: Optional[str] = None,
    save_artifacts: bool = True,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Retrain using best_config.yaml from a sweep with LOCAL artifact logging."""
    sweep_directory = _resolve_input_path(sweep_directory)
    data_input_directory = _resolve_input_path(
        input_directory or os.path.dirname(sweep_directory)
    )
    best_path = os.path.join(sweep_directory, "best_config.yaml")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"best_config.yaml not found in {sweep_directory}")

    cfg = OmegaConf.load(best_path)
    if "tune" in cfg:
        del cfg["tune"]
    cfg = _rebase_config_paths(cfg, data_input_directory, output_directory)
    _artifacts, metrics, output_files = _run_config_experiment(
        cfg,
        output_directory,
        save_artifacts=save_artifacts,
        generate_plots=generate_plots,
    )
    config_out = os.path.join(output_directory, "best_experiment_config.yaml")
    save_experiment_config(cfg, config_out)
    output_files["experiment_config"] = config_out
    result: Dict[str, Any] = {
        "metrics": _extract_flat_metrics(metrics),
        "config_path": config_out,
        "output_files": output_files,
    }
    if "model_artifacts" in output_files:
        result["artifact_dir"] = output_files["model_artifacts"]
    return result


def predict_formulations(
    artifact_directory: str,
    input_directory: str,
    output_directory: str,
    *,
    formulation_path: str,
    component_db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Score new rows with LocalArtifactLogger.load_ensemble_from_artifact."""
    if not formulation_path:
        raise ValueError("formulation_path is required.")
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    artifact_directory = _resolve_input_path(artifact_directory)
    input_directory = _resolve_input_path(input_directory)
    csv_path = _resolve_data_path(formulation_path, input_directory)

    ensemble_model, _X, _Y, _splitter, config, skill_spec = (
        LocalArtifactLogger.load_ensemble_from_artifact(
            artifact_dir=artifact_directory,
            include_training_data=False,
        )
    )

    raw = load_vip_formulation_table(csv_path)
    if detect_data_format(raw)["format"] == "platform_wide":
        df = convert_platform_to_lists(raw)
    else:
        df = load_formulation_csv(csv_path, require_targets=False)

    required_inputs = list(
        OmegaConf.select(config, "data.config.input_columns", default=[]) or []
    )
    missing_inputs = [column for column in required_inputs if column not in df.columns]
    if missing_inputs:
        raise ValueError(
            f"Prediction data is missing required input columns: {missing_inputs}"
        )

    predict_kwargs: Dict[str, Any] = {}
    if component_db_path:
        db_path = _resolve_data_path(component_db_path, input_directory)
        predict_kwargs["component_db"] = _load_component_db_via_dataloader(db_path)
    elif OmegaConf.select(config, "data.config.component_file_path"):
        logging.warning(
            "The training config used a component database but component_db_path "
            "was omitted for prediction. Chemical component processors may fail."
        )

    preds = ensemble_model.predict(df, **predict_kwargs)
    pred_path = os.path.join(output_directory, "predictions.csv")
    if isinstance(preds, pd.DataFrame):
        preds.to_csv(pred_path, index=False)
    else:
        pd.DataFrame(preds).to_csv(pred_path, index=False)

    output_files = {"predictions": pred_path}
    if skill_spec is not None:
        skill_path = os.path.join(output_directory, "skill_specification.json")
        with open(skill_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(skill_spec), handle, indent=2)
        output_files["skill_specification"] = skill_path
    return {
        "n_predictions": int(len(preds)),
        "artifact_directory": artifact_directory,
        "config_model": str(OmegaConf.select(config, "model._target_", default="")),
        "output_files": output_files,
    }


def ml_core_cleanup(deep: bool = False):
    """Clear scratch files created during training."""
    if deep:
        for pattern in ("*.pkl", "*.pickle", "*.csv"):
            for filepath in glob.glob(os.path.join(WORK_DIR, pattern)):
                os.remove(filepath)
    logging.info("Cleanup complete")
