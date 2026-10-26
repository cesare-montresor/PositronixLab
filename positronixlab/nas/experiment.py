import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ax.api.client import Client

from positronixlab.template import TemplateBase
from positronixlab.template.template_variant import TemplateVariant, TemplateVariantList
from positronixlab.trial import TrialConfig
from .storage import get_or_create_client
from .parallel_runner import run_trials_parallel, run_variants_parallel

logger = logging.getLogger(__name__)


class NasExperiment:
    """
    Multi-objective Neural Architecture Search using Meta's modern Ax Client (Ax 1.3+),
    SQLite persistence, and 2-phase variant bootstrapping:
    1. Phase 1: Parallel exploration across common unconstrained template space.
    2. Phase 2: Concurrent variant Bayesian optimizations, each with its own SQLite database,
       passing variant fixed_features to client.get_next_trials to freeze those search space dimensions.
    3. Resumability: Checks existing databases and resumes uninterrupted from the last completed trial.
    4. Pareto Frontier: Extracts best trade-offs using Ax's native get_best_parameterization().
    5. Folder Structure (always rooted at output/):
       output/<exp_name>/
       ├── common/
       │   ├── trials/
       │   └── best/
       ├── <variant_name>/
       │   ├── trials/
       │   └── best/
       └── databases/
    """

    def __init__(
        self,
        name: str,
        template: TemplateBase,
        eval_fn: Callable[[TrialConfig], Dict[str, float]],
        objectives: str = "val_acc, -model_size, modularity_score",
        outcome_constraints: Optional[Sequence[str]] = None,
        variants: Optional[TemplateVariantList] = None,
        n_common_trials: int = 8,
        n_variant_trials: int = 6,
        parallel_common: int = 4,
        parallel_variants: int = 4,
        output_dir: str = "output"
    ):
        self.name = name
        self.template = template
        self.eval_fn = eval_fn
        self.objectives = objectives
        self.outcome_constraints = list(outcome_constraints or [])
        self.n_common_trials = n_common_trials
        self.n_variant_trials = n_variant_trials
        self.parallel_common = parallel_common
        self.parallel_variants = parallel_variants

        # Structured output directories: output/<exp_name>/<variant>/[trials, best]
        self.exp_root_dir = os.path.abspath(os.path.join(output_dir, self.name))
        self.db_dir = os.path.abspath(os.path.join(self.exp_root_dir, "databases"))
        os.makedirs(self.exp_root_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)

        self.variants = variants if variants is not None else self.template.variants
        self.variants.validate_all(self.template)

        self.common_client: Optional[Client] = None
        self.variant_clients: Dict[str, Client] = {}

    def _get_variant_dirs(self, variant_name: str) -> Dict[str, str]:
        """Creates and returns the structured folders: output/exp_name/variant/trials and best."""
        var_dir = os.path.join(self.exp_root_dir, variant_name)
        trials_dir = os.path.join(var_dir, "trials")
        best_dir = os.path.join(var_dir, "best")
        os.makedirs(trials_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)
        return {
            "root": var_dir,
            "trials": trials_dir,
            "best": best_dir,
        }

    # -------------------------------------------------------------------------
    # Phase 1: Common Phase (Random Search with Parallel Execution)
    # -------------------------------------------------------------------------
    def run_common_phase(self) -> List[Dict[str, Any]]:
        dirs = self._get_variant_dirs("common")
        db_path = os.path.join(self.db_dir, f"{self.name}_common.db")

        # Configure Client with full template search space
        self.common_client = get_or_create_client(
            experiment_name=f"{self.name}_common",
            db_path=db_path,
            parameters=self.template.to_parameter_configs(),
            objectives=self.objectives,
            outcome_constraints=self.outcome_constraints,
        )

        completed_count = len(self.common_client.experiment.trials)
        logger.info(f"Phase 1 (Common): {completed_count}/{self.n_common_trials} trials already completed.")

        fixed_params = self.template.get_fixed()

        def _evaluate_wrapper(params: Dict[str, Any], trial_index: int) -> Dict[str, float]:
            config = TrialConfig.from_params_and_fixed(
                params=params,
                fixed=fixed_params,
                variant_name="common",
                trial_index=trial_index,
            )
            # Save trial config into output/exp_name/common/trials/
            config.save(os.path.join(dirs["trials"], f"trial_{trial_index:03d}.yaml"))
            return self.eval_fn(config)

        while completed_count < self.n_common_trials:
            batch_size = min(self.parallel_common, self.n_common_trials - completed_count)
            trials_batch = self.common_client.get_next_trials(max_trials=batch_size)

            results = run_trials_parallel(
                trial_batch=trials_batch,
                eval_fn=_evaluate_wrapper,
                max_workers=self.parallel_common,
            )

            for t_idx, metrics in results:
                self.common_client.complete_trial(trial_index=t_idx, raw_data=metrics)
                logger.info(f"Common Trial #{t_idx} completed: {metrics}")

            completed_count = len(self.common_client.experiment.trials)

        # Save best configs in common phase to output/exp_name/common/best/
        try:
            best_pareto = self.common_client.get_best_parameterization()
            for idx, item in enumerate(best_pareto):
                params = item[0] if isinstance(item, tuple) else (item.get("parameters") if isinstance(item, dict) else item)
                t_idx = item[2] if isinstance(item, tuple) and len(item) > 2 else idx
                best_config = TrialConfig.from_params_and_fixed(
                    params=params,
                    fixed=fixed_params,
                    variant_name="common",
                    trial_index=t_idx,
                )
                best_config.save(os.path.join(dirs["best"], f"best_{t_idx:03d}.yaml"))
        except Exception as e:
            logger.debug(f"Common best parameterization export skipped: {e}")

        # Collect evaluated trials for bootstrapping
        prior_trials = []
        for t_idx, trial in self.common_client.experiment.trials.items():
            if trial.status.is_completed:
                params = trial.arm.parameters
                metrics = {k: v[0] for k, v in trial.raw_data.items()} if hasattr(trial, "raw_data") and trial.raw_data else {}
                prior_trials.append({"trial_index": t_idx, "parameters": params, "metrics": metrics})

        return prior_trials

    # -------------------------------------------------------------------------
    # Phase 2: Variant Bayesian Phases (Bootstrapped & Concurrent)
    # -------------------------------------------------------------------------
    def _run_single_variant(self, variant: TemplateVariant, prior_trials: List[Dict[str, Any]]) -> Client:
        v_name = variant.name
        dirs = self._get_variant_dirs(v_name)
        db_path = os.path.join(self.db_dir, f"{v_name}.db")

        # Client configured with full template search space
        client = get_or_create_client(
            experiment_name=f"{self.name}_{v_name}",
            db_path=db_path,
            parameters=self.template.to_parameter_configs(),
            objectives=self.objectives,
            outcome_constraints=self.outcome_constraints,
        )

        variant_fixed_features = variant.get_fixed_features()
        fixed_params = self.template.get_fixed()

        # 1. Bootstrap prior trials if starting fresh
        if len(client.experiment.trials) == 0:
            logger.info(f"[{v_name}] Bootstrapping {len(prior_trials)} prior observations...")
            for prior in prior_trials:
                adapted_params = dict(prior["parameters"])
                attach_res = client.attach_trial(parameters=adapted_params)
                t_idx = attach_res[0] if isinstance(attach_res, tuple) else attach_res
                if prior["metrics"]:
                    client.complete_trial(trial_index=t_idx, raw_data=prior["metrics"])

        # 2. Sequential Bayesian optimization steps freezing variant_fixed_features via get_next_trials
        current_trials = len(client.experiment.trials)
        target_trials = current_trials + self.n_variant_trials
        logger.info(f"[{v_name}] Running BO steps up to {target_trials} trials (frozen features: {variant_fixed_features})...")

        # Prepare fixed_features for Client.get_next_trials per https://ax.readthedocs.io/en/stable/api.html#ax.api.client.Client.get_next_trials
        try:
            from ax.core.observation import ObservationFeatures
            fixed_feats_arg = ObservationFeatures(parameters=variant_fixed_features)
        except Exception:
            fixed_feats_arg = variant_fixed_features

        while len(client.experiment.trials) < target_trials:
            # Pass variant fixed_features into get_next_trials
            try:
                trials_dict = client.get_next_trials(max_trials=1, fixed_features=fixed_feats_arg)
            except (TypeError, Exception):
                trials_dict = client.get_next_trials(max_trials=1, fixed_features=variant_fixed_features)

            t_idx, params = next(iter(trials_dict.items()))

            # Ensure frozen features are present in params
            full_params = dict(variant_fixed_features)
            full_params.update(params)

            # Initialize TrialConfig with params (including frozen features) and template fixed
            config = TrialConfig.from_params_and_fixed(
                params=full_params,
                fixed=fixed_params,
                variant_name=v_name,
                trial_index=t_idx,
            )
            # Save into output/exp_name/variant/trials/
            config.save(os.path.join(dirs["trials"], f"trial_{t_idx:03d}.yaml"))

            metrics = self.eval_fn(config)
            client.complete_trial(trial_index=t_idx, raw_data=metrics)
            logger.info(f"[{v_name}] BO Trial #{t_idx} completed: {metrics}")

        # 3. Save Pareto-optimal configurations to output/exp_name/variant/best/
        try:
            best_pareto = client.get_best_parameterization()
            for idx, item in enumerate(best_pareto):
                params = item[0] if isinstance(item, tuple) else (item.get("parameters") if isinstance(item, dict) else item)
                t_idx = item[2] if isinstance(item, tuple) and len(item) > 2 else idx
                full_params = dict(variant_fixed_features)
                full_params.update(params)

                best_config = TrialConfig.from_params_and_fixed(
                    params=full_params,
                    fixed=fixed_params,
                    variant_name=v_name,
                    trial_index=t_idx,
                )
                best_config.save(os.path.join(dirs["best"], f"best_{t_idx:03d}.yaml"))
        except Exception as e:
            logger.debug(f"[{v_name}] Pareto best parameterization export error: {e}")

        return client

    def run_variant_phases(self, prior_trials: List[Dict[str, Any]]) -> Dict[str, Client]:
        tasks = [
            (
                variant.name,
                lambda v=variant: self._run_single_variant(v, prior_trials),
            )
            for variant in self.variants
        ]

        self.variant_clients = run_variants_parallel(
            variant_tasks=tasks,
            max_workers=self.parallel_variants,
        )
        return self.variant_clients

    # -------------------------------------------------------------------------
    # Pareto Best Extraction via Ax Native get_best_parameterization()
    # -------------------------------------------------------------------------
    def get_pareto_best(self) -> Dict[str, List[TrialConfig]]:
        """
        Uses Ax's native client.get_best_parameterization() to extract the Pareto frontier
        for each variant and saves configs to output/exp_name/variant/best/.
        """
        results: Dict[str, List[TrialConfig]] = {}
        fixed_params = self.template.get_fixed()

        for v_name, client in self.variant_clients.items():
            dirs = self._get_variant_dirs(v_name)
            variant = self.variants.get_variant(v_name)
            variant_fixed_features = variant.get_fixed_features() if variant else {}
            results[v_name] = []

            best_pareto = client.get_best_parameterization()
            for idx, item in enumerate(best_pareto):
                if isinstance(item, tuple):
                    params = item[0]
                    t_idx = item[2] if len(item) > 2 else idx
                elif isinstance(item, dict):
                    params = item.get("parameters", item)
                    t_idx = item.get("trial_index", idx)
                else:
                    continue

                full_params = dict(variant_fixed_features)
                full_params.update(params)

                config = TrialConfig.from_params_and_fixed(
                    params=full_params,
                    fixed=fixed_params,
                    variant_name=v_name,
                    trial_index=t_idx,
                )
                config.save(os.path.join(dirs["best"], f"pareto_best_{t_idx:03d}.yaml"))
                results[v_name].append(config)

        # Write overall summary to output/exp_root_dir/summary.json
        summary = {
            "experiment_name": self.name,
            "template": self.template.name,
            "objectives": self.objectives,
            "fixed_parameters": fixed_params,
            "variants": {
                v_name: {
                    "fixed_features": self.variants.get_variant(v_name).get_fixed_features(),
                    "num_pareto_best": len(configs),
                }
                for v_name, configs in results.items()
            },
            "timestamp": time.time(),
        }
        with open(os.path.join(self.exp_root_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        return results

    # -------------------------------------------------------------------------
    # Full Execution Pipeline
    # -------------------------------------------------------------------------
    def run(self) -> Dict[str, List[TrialConfig]]:
        logger.info(f"Starting Multi-Objective NAS Experiment '{self.name}'...")
        logger.info(f"Results output root directory: {self.exp_root_dir}")

        # Phase 1: Common Random Search
        prior_trials = self.run_common_phase()

        # Phase 2: Concurrent Variant Bayesian Optimization
        self.run_variant_phases(prior_trials)

        # Pareto Extraction via native Ax MOO
        pareto_configs = self.get_pareto_best()
        logger.info(f"Experiment completed. Summary and configs saved to: {self.exp_root_dir}")
        return pareto_configs
