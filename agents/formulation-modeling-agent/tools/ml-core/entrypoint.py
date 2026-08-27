#!/usr/bin/env python3
"""Deterministic action entrypoint for the ml-core Discovery tool.

Invoked by tool.yaml actions:
  python3 /app/entrypoint.py --action <name> --input ... --output ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from typing import Any, Callable, Dict, Optional

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_core_utils import (  # noqa: E402
    predict_formulations,
    quick_finish,
    quick_setup,
    run_experiment_action,
    run_hyperparameter_sweep,
    save_final_results,
    train_best_from_sweep,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ml_core_entrypoint")


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_results(results: Dict[str, Any], output_dir: str) -> None:
    output_files = dict(results.get("output_files") or {})
    summary = {k: v for k, v in results.items() if k != "output_files"}
    import ml_core_utils

    ml_core_utils.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_final_results(summary, output_files)
    results_path = os.path.join(output_dir, "action_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


ActionFn = Callable[..., Dict[str, Any]]


def action_run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.config_path:
        raise ValueError(
            "config_path is required for run_experiment. "
            "Author experiment_config.yaml in the coding environment first."
        )
    return run_experiment_action(
        args.input,
        args.output,
        config_path=args.config_path,
        save_artifacts=_parse_bool(args.save_artifacts, True),
        generate_plots=_parse_bool(args.generate_plots, True),
    )


def action_run_hyperparameter_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.config_path:
        raise ValueError(
            "config_path is required for run_hyperparameter_sweep. "
            "Author sweep_config.yaml (including a tune block) in the coding environment first."
        )
    return run_hyperparameter_sweep(
        args.input,
        args.output,
        config_path=args.config_path,
    )


def action_train_best_from_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    sweep_dir = args.sweep_directory or args.input
    return train_best_from_sweep(
        sweep_dir,
        args.output,
        input_directory=args.input,
        save_artifacts=_parse_bool(args.save_artifacts, True),
        generate_plots=_parse_bool(args.generate_plots, True),
    )


def action_predict(args: argparse.Namespace) -> Dict[str, Any]:
    artifact_dir = args.artifact_directory
    if not artifact_dir:
        raise ValueError("--artifact-directory is required for predict")
    if not args.formulation_path:
        raise ValueError("formulation_path is required for predict")
    component_db_path = (getattr(args, "component_db_path", None) or "").strip() or None
    return predict_formulations(
        artifact_dir,
        args.input,
        args.output,
        formulation_path=args.formulation_path,
        component_db_path=component_db_path,
    )


ACTIONS: Dict[str, ActionFn] = {
    "run_experiment": action_run_experiment,
    "run_hyperparameter_sweep": action_run_hyperparameter_sweep,
    "train_best_from_sweep": action_train_best_from_sweep,
    "predict": action_predict,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ml-core Discovery tool entrypoint")
    parser.add_argument(
        "--action",
        required=True,
        choices=sorted(ACTIONS.keys()),
        help="Action to run",
    )
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to experiment_config.yaml or sweep_config.yaml (required for train/sweep)",
    )
    parser.add_argument("--save-artifacts", default=None)
    parser.add_argument("--generate-plots", default=None)
    parser.add_argument("--sweep-directory", default=None)
    parser.add_argument("--artifact-directory", default=None)
    parser.add_argument(
        "--formulation-path",
        default=None,
        help="Path to the formulation CSV to score (required for predict)",
    )
    parser.add_argument(
        "--component-db-path",
        default=None,
        help="Optional path to a component database CSV for predict",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    work_dir = "/workdir" if os.path.isdir("/workdir") or os.access("/", os.W_OK) else output_dir
    try:
        os.makedirs(work_dir, exist_ok=True)
    except OSError:
        work_dir = output_dir
        os.makedirs(work_dir, exist_ok=True)

    quick_setup(input_dir=input_dir, output_dir=output_dir, work_dir=work_dir)
    logger.info("Running action=%s input=%s output=%s", args.action, input_dir, output_dir)

    try:
        handler = ACTIONS[args.action]
        results = handler(args)
        _write_results(results, output_dir)
        logger.info("Action %s completed successfully", args.action)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Action %s failed: %s", args.action, exc)
        traceback.print_exc()
        _write_results(
            {
                "status": "failed",
                "error": str(exc),
                "action": args.action,
                "output_files": {},
            },
            output_dir,
        )
        return 1
    finally:
        quick_finish()


if __name__ == "__main__":
    raise SystemExit(main())
