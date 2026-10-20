"""Template-driven Ax experimentation.

Ax owns experiment state and persistence.  This module only adapts the
ModularNet template schema to Ax and coordinates the common and variant
phases.
"""

from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class AxObjective:
    name: str
    minimize: bool = False
    threshold: float | None = None


@dataclass(frozen=True)
class AxPhaseResult:
    phase: str
    variant: str | None
    trial_ids: tuple[int, ...]


@dataclass(frozen=True)
class AxExperimentResult:
    experiment: str
    phases: tuple[AxPhaseResult, ...]
    best_trials: tuple[dict[str, Any], ...]
    pareto_trials: tuple[dict[str, Any], ...]


Evaluator = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, float],
]


def _ax_api():
    try:
        from ax.api.client import Client
        from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig, StorageConfig
    except ImportError as exc:
        raise ImportError(
            "NAS requires ax-platform with the modern ax.api.client API."
        ) from exc
    return Client, ChoiceParameterConfig, RangeParameterConfig, StorageConfig


def _objectives(raw: Mapping[str, str | AxObjective]) -> tuple[AxObjective, ...]:
    objectives = tuple(
        value if isinstance(value, AxObjective)
        else AxObjective(name=name, minimize=value == "minimize")
        for name, value in raw.items()
    )
    if not objectives:
        raise ValueError("At least one objective is required")
    return objectives


def _parameter_configs(template: Any, fixed: Mapping[str, Any] | None = None):
    _, ChoiceParameterConfig, RangeParameterConfig, _ = _ax_api()
    fixed = fixed or {}
    configs = []
    for definition in template.params.values():
        name = definition["name"]
        if definition["type"] == "fixed" or name in fixed:
            continue
        parameter_type = definition["value_type"]
        if definition["type"] == "range":
            configs.append(
                RangeParameterConfig(
                    name=name,
                    bounds=tuple(definition["bounds"]),
                    parameter_type=parameter_type,
                    scaling="log" if definition.get("log_scale") else None,
                )
            )
        elif definition["type"] == "choice":
            configs.append(
                ChoiceParameterConfig(
                    name=name,
                    values=list(definition["values"]),
                    parameter_type=parameter_type,
                    is_ordered=definition.get("is_ordered"),
                    dependent_parameters=definition.get("dependents"),
                )
            )
        else:
            raise ValueError(f"Unsupported template parameter type: {definition['type']}")
    return configs


def _config(template: Any, parameters: Mapping[str, Any], fixed: Mapping[str, Any]):
    values = {
        name: definition.get("value", definition.get("values", [None])[0])
        for name, definition in template.params.items()
        if definition["type"] == "fixed"
    }
    values.update(parameters)
    values.update(fixed)
    return values


class AxExperiment:
    """Run a parallel common phase and independent sequential variants.

    The SQLite file is managed by Ax. Each phase is a named Ax experiment in
    that database, so it can be restored with ``Client.load_from_database``.
    """

    def __init__(
        self,
        template: Any,
        db_path: str | Path,
        objectives: Mapping[str, str | AxObjective],
        name: str = "experiment",
        evaluator: Evaluator | None = None,
        executor: Executor | None = None,
        max_parallelism: int = 1,
        client_factory: Callable[..., Any] | None = None,
    ):
        self.template = template
        self.db_path = Path(db_path)
        self.objectives = _objectives(objectives)
        self.name = name
        self.evaluator = evaluator
        self.executor = executor
        self.max_parallelism = max(1, max_parallelism)
        self._client_factory = client_factory
        self._clients: dict[str, Any] = {}
        self._generation_ready: set[str] = set()
        self._observations: dict[
            str, list[tuple[dict[str, Any], dict[str, float], dict[str, Any]]]
        ] = {}

    def _storage_config(self):
        _, _, _, StorageConfig = _ax_api()
        return StorageConfig(url=f"sqlite:///{self.db_path.resolve()}")

    def _new_client(self, phase: str, fixed: Mapping[str, Any] | None = None):
        Client, _, _, _ = _ax_api()
        experiment_name = f"{self.name}_{phase}"
        storage_config = self._storage_config()
        try:
            from ax.exceptions.core import ObjectNotFoundError
            client = Client.load_from_database(
                experiment_name=experiment_name,
                storage_config=storage_config,
            )
            self._generation_ready.add(phase)
        except ObjectNotFoundError:
            client = (self._client_factory(storage_config=storage_config)
                      if self._client_factory else Client(storage_config=storage_config))
            client.configure_experiment(
                parameters=_parameter_configs(self.template, fixed),
                name=experiment_name,
            )
            objective = ", ".join(
                ("-" if item.minimize else "") + item.name for item in self.objectives
            )
            client.configure_optimization(objective=objective)
        self._clients[phase] = client
        return client

    def _evaluate(self, phase, variant, trial_id, parameters, fixed):
        if self.evaluator is None:
            raise ValueError("An evaluator callback is required to run an experiment")
        config = _config(self.template, parameters, fixed)
        values = dict(self.evaluator(config, {
            "experiment": self.name,
            "phase": phase,
            "variant": variant,
            "ax_trial_id": trial_id,
        }))
        missing = {objective.name for objective in self.objectives} - values.keys()
        if missing:
            raise ValueError(f"Evaluator did not return objectives: {sorted(missing)}")
        return trial_id, values, config, dict(parameters)

    def _complete(self, client, result):
        trial_id, values, _, _ = result
        client.complete_trial(trial_index=trial_id, raw_data=values)

    def _run_phase(self, client, phase, variant, trials, fixed, parallelism):
        if phase not in self._generation_ready:
            client.configure_generation_strategy(
                method="random_search" if phase == "common" else "fast",
                use_existing_trials_for_initialization=True,
            )
            self._generation_ready.add(phase)
        executor = self.executor or ThreadPoolExecutor(max_workers=parallelism)
        owned = self.executor is None
        trial_ids = []
        observations = self._observations.setdefault(phase, [])
        try:
            remaining = trials
            while remaining:
                batch_size = min(parallelism, remaining)
                generated = client.get_next_trials(max_trials=batch_size)
                remaining -= len(generated)
                pending = [
                    executor.submit(
                        self._evaluate, phase, variant, trial_id, parameters, fixed
                    )
                    for trial_id, parameters in generated.items()
                ]
                for future in as_completed(pending):
                    result = future.result()
                    self._complete(client, result)
                    trial_ids.append(result[0])
                    observations.append((result[3], result[1], result[2]))
        finally:
            if owned:
                executor.shutdown(wait=True)
        return AxPhaseResult(phase, variant, tuple(trial_ids))

    def _seed_variant(self, client, observations, fixed):
        for parameters, values, config in observations:
            if not all(config.get(name) == value for name, value in fixed.items()):
                continue
            reduced = {
                name: value for name, value in parameters.items()
                if name not in fixed
            }
            trial_id = client.attach_trial(parameters=reduced)
            client.complete_trial(trial_index=trial_id, raw_data=values)

    def run(
        self,
        trials: int,
        variants: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> AxExperimentResult:
        if trials < 1:
            raise ValueError("trials must be positive")
        common = self._new_client("common")
        common_result = self._run_phase(
            common, "common", None, trials, {}, self.max_parallelism
        )
        observations = self._observations["common"]

        variant_results = []
        for name, fixed in (variants or {}).items():
            client = self._new_client(name, fixed)
            self._seed_variant(client, observations, fixed)
            variant_results.append(
                self._run_phase(client, name, name, trials, fixed, 1)
            )

        return AxExperimentResult(
            experiment=self.name,
            phases=(common_result, *variant_results),
            best_trials=tuple(self.best_trials()),
            pareto_trials=tuple(self.pareto_trials()),
        )

    def best_trials(self) -> list[dict[str, Any]]:
        """Return the best observed point from each single-objective phase."""
        if len(self.objectives) != 1:
            return []
        result = []
        for phase, client in self._clients.items():
            trial_id, parameters, prediction = client.get_best_parameterization(
                use_model_predictions=False
            )
            result.append({
                "phase": phase,
                "trial_id": trial_id,
                "parameters": parameters,
                "objectives": prediction,
            })
        return result

    def pareto_trials(self) -> list[dict[str, Any]]:
        """Return Ax's observed multi-objective Pareto frontier."""
        if len(self.objectives) < 2:
            return []
        result = []
        for phase, client in self._clients.items():
            for parameters, values, trial_id, arm_name in client.get_pareto_frontier(
                use_model_predictions=False
            ):
                result.append({
                    "phase": phase,
                    "trial_id": trial_id,
                    "arm_name": arm_name,
                    "parameters": parameters,
                    "objectives": values,
                })
        return result
