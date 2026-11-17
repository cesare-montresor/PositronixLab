import logging
import time
from typing import Any, Callable, Dict, Optional

from positronixlab.pipeline import TrialManager, PipelineConfig

logger = logging.getLogger(__name__)


class AtlasComposeTrialEvaluator:
    """Evaluates trials using the AtlasCompose PluginManager and worker pipeline."""

    def __init__(self, trial_class: Any, timeout_sec: int = 600):
        self.trial_class = trial_class
        self.timeout_sec = timeout_sec

    def __call__(
        self,
        parameters: Dict[str, Any],
        trial_index: int,
        fixed: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None,
    ) -> Dict[str, float]:
        config = PipelineConfig.from_params_and_fixed(
            params=parameters,
            fixed=fixed,
            variant_name=variant_name,
            trial_index=trial_index,
        )

        trial_manager = TrialManager()
        trial = self.trial_class(config, trial_manager)

        logger.info(f"Starting PluginManager trial #{trial_index} ({variant_name or 'common'})...")
        trial.run()

        start_time = time.time()
        while time.time() - start_time < self.timeout_sec:
            if trial.is_complete():
                break
            time.sleep(0.5)
        else:
            trial.shutdown()
            raise TimeoutError(f"Trial #{trial_index} exceeded timeout of {self.timeout_sec}s")

        all_vars = trial.plugin_manager.env_get()
        trial.shutdown()

        # Extract objective metrics
        metrics = {
            "val_acc": float(all_vars.get("val_acc", all_vars.get("accuracy", 0.0))),
            "model_size": float(all_vars.get("model_size", 0.0)),
        }
        if "modularity_score" in all_vars:
            metrics["modularity_score"] = float(all_vars["modularity_score"])

        return metrics
