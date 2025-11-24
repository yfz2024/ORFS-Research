#!/usr/bin/env python3
"""
Utility to run stage_optimize.py with explicit per-stage parameter overrides.

Overrides can be provided via JSON files or inline CLI expressions of the form
`stage.parameter=value`. Parameter tokens may use either the logical parameter
keys (e.g. `core_util`) or their corresponding environment variables
(`CORE_UTILIZATION`). Values are coerced using JSON parsing semantics, so bare
numbers/booleans/null work without quoting while opaque strings can be wrapped
in quotes if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from stage_optimize import (
    PARAM_TO_ENV_VAR,
    StageOptimizationDriver,
    StageDefinition,
    _stage_definitions,
)


StageOverrideMap = Dict[str, Dict[str, Any]]


def _build_stage_lookup() -> Dict[str, StageDefinition]:
    return {stage.name.lower(): stage for stage in _stage_definitions()}


def _stage_env_vars(stage: StageDefinition) -> Dict[str, None]:
    envs: Dict[str, None] = {}
    for key in stage.parameter_keys:
        env_var = PARAM_TO_ENV_VAR.get(key)
        if env_var:
            envs[env_var] = None
    return envs


def _build_env_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for key, env_var in PARAM_TO_ENV_VAR.items():
        aliases[key.lower()] = env_var
        aliases[env_var.lower()] = env_var
    return aliases


STAGE_LOOKUP = _build_stage_lookup()
ENV_ALIASES = _build_env_aliases()


def _resolve_stage(name: str) -> StageDefinition:
    lookup_key = name.strip().lower()
    if lookup_key not in STAGE_LOOKUP:
        raise ValueError(f"Unknown stage '{name}'. Valid stages: {', '.join(sorted(STAGE_LOOKUP))}")
    return STAGE_LOOKUP[lookup_key]


def _normalize_param(token: str) -> str:
    if not token:
        raise ValueError("Empty parameter override key encountered.")
    alias = ENV_ALIASES.get(token.strip().lower())
    if alias is None:
        valid = ", ".join(sorted(set(ENV_ALIASES.values())))
        raise ValueError(f"Unknown parameter '{token}'. Valid options include: {valid}")
    return alias


def _parse_scalar(raw_value: str) -> Any:
    text = raw_value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _load_file_overrides(path: Path) -> StageOverrideMap:
    if not path.is_file():
        raise ValueError(f"Override file '{path}' does not exist.")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse overrides file '{path}': {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Overrides file '{path}' must contain a JSON object at the top level.")

    return _normalize_stage_mapping(payload, source=str(path))


def _normalize_stage_mapping(data: Mapping[str, Any], *, source: str) -> StageOverrideMap:
    overrides: StageOverrideMap = {}
    for stage_name, params in data.items():
        stage = _resolve_stage(stage_name)
        stage_envs = _stage_env_vars(stage)
        if not params:
            continue
        if not isinstance(params, Mapping):
            raise ValueError(
                f"{source}: stage '{stage.name}' overrides must be an object of ENV=value entries."
            )
        stage_overrides = overrides.setdefault(stage.name, {})
        for param_key, value in params.items():
            env_var = _normalize_param(param_key)
            if env_var not in stage_envs:
                valid = ", ".join(stage_envs.keys()) or "(no adjustable parameters)"
                raise ValueError(
                    f"{source}: parameter '{env_var}' is not adjustable for stage '{stage.name}'. "
                    f"Valid parameters: {valid}"
                )
            stage_overrides[env_var] = value
    return overrides


def _parse_inline_overrides(values: Any) -> StageOverrideMap:
    overrides: StageOverrideMap = {}
    for expr in values or []:
        if "=" not in expr:
            raise ValueError(
                f"Invalid inline override '{expr}'. Expected format 'stage.param=value'."
            )
        lhs, rhs = expr.split("=", 1)
        if "." not in lhs:
            raise ValueError(
                f"Invalid inline override '{expr}'. Expected 'stage.param=value' format."
            )
        stage_token, param_token = lhs.split(".", 1)
        stage = _resolve_stage(stage_token)
        env_var = _normalize_param(param_token)

        stage_envs = _stage_env_vars(stage)
        if env_var not in stage_envs:
            valid = ", ".join(stage_envs.keys()) or "(no adjustable parameters)"
            raise ValueError(
                f"Inline override '{expr}' targets '{env_var}' which is not adjustable during stage "
                f"'{stage.name}'. Valid parameters: {valid}"
            )

        value = _parse_scalar(rhs)
        overrides.setdefault(stage.name, {})[env_var] = value
    return overrides


def _merge_overrides(target: StageOverrideMap, addition: StageOverrideMap) -> None:
    for stage, params in addition.items():
        if not params:
            continue
        target.setdefault(stage, {}).update(params)


def run_with_overrides(args: argparse.Namespace) -> None:
    overrides: StageOverrideMap = {}

    if args.overrides_file:
        for file_path in args.overrides_file:
            file_overrides = _load_file_overrides(file_path)
            _merge_overrides(overrides, file_overrides)

    inline_overrides = _parse_inline_overrides(args.override)
    _merge_overrides(overrides, inline_overrides)

    if not overrides:
        raise ValueError("No overrides provided. Use --overrides-file or --override.")

    print("Loaded stage overrides:")
    for stage in sorted(overrides):
        for env_var, value in overrides[stage].items():
            print(f"  {stage}.{env_var} = {value}")

    driver = StageOptimizationDriver(
        platform=args.platform,
        design=args.design,
        objective=args.objective,
        stages=args.stages,
        max_react_steps=args.max_react_steps,
        temperature=args.temperature,
        dry_run=args.dry_run,
        workdir=args.workdir,
    )
    driver.stage_overrides = overrides
    driver.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply explicit parameter overrides when running stage_optimize.py."
    )
    parser.add_argument("platform", help="PDK/platform name (e.g. asap7, sky130hd).")
    parser.add_argument("design", help="Design name (e.g. aes, ibex).")
    parser.add_argument("objective", help="Optimization objective (ECP, DWL, COMBO).")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Subset of stages to run (default: all). Options: synth floorplan place cts route finish.",
    )
    parser.add_argument(
        "--max-react-steps",
        type=int,
        default=3,
        help="Maximum reasoning steps for the ReAct loop.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the make commands without executing them.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Flow workspace root (default: current working directory).",
    )
    parser.add_argument(
        "-f",
        "--overrides-file",
        action="append",
        type=Path,
        help="Path to a JSON file describing stage overrides.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Inline override in the form 'stage.parameter=value'. Repeatable.",
    )
    return parser


def main(argv: Any = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_with_overrides(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
