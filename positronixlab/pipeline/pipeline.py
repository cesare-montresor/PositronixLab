from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Dict, List, Optional


from parallelazy import PluginManager, is_debugpy_attached

from positronixlab.pipeline.pipeline_config import PipelineConfig, PipelineOptions


logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """
    Generic base class for `parallelazy`-backed pipelines.

    Separates `config` (`PipelineConfig`: shared hyperparameters published to
    the plugin environment as `"pipeline_config"`) from `options`
    (`PipelineOptions(ManagerOptions)`: `PluginManager` & pipeline runtime controls).
    """

    def __init__(
        self,
        config: PipelineConfig,
        options: Optional[PipelineOptions] = None,
    ):
        self.config = config
        self.options = options or PipelineOptions()

        self.plugin_manager = PluginManager(**self.options.to_manager_kwargs())
        self.plugin_manager.worker_pydebug = is_debugpy_attached()
        self.plugin_manager.env_set("pipeline_config", self.config)

        self._closed: bool = False

    @property
    def work_dir(self) -> str:
        return self.plugin_manager.work_dir

    @property
    def log_dir(self) -> str:
        return self.plugin_manager.log_dir

    @property
    def run_log_dir(self) -> str:
        return self.plugin_manager.run_log_dir

    def enable_monitoring(self, hotkey: Optional[str] = "m") -> None:
        """Enable `parallelazy` runtime monitoring (and optional report hotkey)."""
        self.plugin_manager.enable_monitoring()
        if hotkey:
            self.plugin_manager.enable_report_hotkey(hotkey)

    def run(self, parallel: Optional[bool] = None) -> None:
        """Start all loaded plugins."""
        use_parallel = self.options.parallel_plugins if parallel is None else parallel
        self.plugin_manager.start_all_plugins(parallel=use_parallel)

    def get_env(self, key: Optional[str] = None) -> Any:
        """Read a key (or the full dict when `key=None`) from the shared plugin environment."""
        return self.plugin_manager.env_get(key) if key is not None else self.plugin_manager.env_get()

    def all_plugins_terminated(self) -> bool:
        """Return True if plugins were loaded and none are alive anymore."""
        statuses = self.plugin_manager.plugin_status_all()
        return bool(statuses) and not any(s.process_alive for s in statuses.values())

    @abstractmethod
    def is_complete(self) -> bool:
        """Check whether this pipeline has finished executing."""
        pass

    def is_error(self) -> bool:
        """Return True if any loaded plugin worker is in an error state."""
        statuses = self.plugin_manager.plugin_status_all()
        for status in statuses.values():
            if status.worker_status and status.worker_status.state == "error":
                return True
        return False

    def get_errors(self) -> Dict[str, List[str]]:
        """
        Collect error messages across all loaded plugins.

        Reads `worker_status.errors` from `plugin_status_all()` first (fast and
        works even if the worker process already died), falling back to
        `plugin_errors(plugin_id)` when alive.
        """
        errors: Dict[str, List[str]] = {}
        statuses = self.plugin_manager.plugin_status_all()
        for plugin_id, status in statuses.items():
            errs: List[str] = []
            raw_errs = getattr(status.worker_status, "errors", None) if status.worker_status else None
            if isinstance(raw_errs, (list, tuple)) and raw_errs:
                errs = [str(e) for e in raw_errs]
            elif status.process_alive:
                try:
                    fetched = self.plugin_manager.plugin_errors(plugin_id)
                    if fetched:
                        errs = [str(e) for e in fetched]
                except Exception:
                    logger.debug("Failed to fetch plugin_errors for %s", plugin_id, exc_info=True)
            if not errs and isinstance(raw_errs, int) and raw_errs > 0:
                errs = [f"Worker reported {raw_errs} error(s)"]
            if errs:
                errors[plugin_id] = errs
        return errors

    def wait(
        self,
        timeout_sec: Optional[float] = None,
        poll_interval: Optional[float] = None,
        terminate_on_error: Optional[bool] = None,
        status_log_interval: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """
        Block until `is_complete()` is True, an error occurs, or `timeout_sec` elapses.

        All arguments default to the corresponding fields on `self.options`.
        `timeout_sec=None` waits indefinitely (useful for long runs; set for debugging).
        """
        timeout = self.options.timeout_sec if timeout_sec is None else timeout_sec
        interval = self.options.poll_interval if poll_interval is None else poll_interval
        stop_on_err = self.options.terminate_on_error if terminate_on_error is None else terminate_on_error
        log_interval = self.options.status_log_interval if status_log_interval is None else status_log_interval
        is_verbose = self.options.verbose if verbose is None else verbose

        t0 = time.monotonic()
        last_log = t0

        while True:
            now = time.monotonic()
            if log_interval is not None and (now - last_log) >= log_interval:
                self._log_status_snapshot(elapsed=now - t0, verbose=is_verbose)
                last_log = now

            if self.is_error():
                errors = self.get_errors()
                flat = [msg for msgs in errors.values() for msg in msgs]
                detail = "\n".join(flat) if flat else "Plugin worker entered error state."
                if stop_on_err:
                    self.shutdown()
                    raise RuntimeError(detail)
                logger.error("Plugin error detected (terminate_on_error=False): %s", detail)

            if self.is_complete():
                return

            if timeout is not None and (now - t0) >= timeout:
                self.shutdown()
                raise TimeoutError(f"Pipeline exceeded timeout of {timeout}s")

            time.sleep(interval)

    def _log_status_snapshot(self, elapsed: float, verbose: bool) -> None:
        statuses = self.plugin_manager.plugin_status_all()
        if verbose:
            logger.info("Pipeline status check (%.1fs):", elapsed)
        for plugin_id, status in statuses.items():
            has_problem = False
            if status.worker_status:
                if getattr(status.worker_status, "errors", None):
                    has_problem = True
                    msg = f"[{plugin_id}] ERRORS: {status.worker_status.errors}"
                else:
                    msg = f"[{plugin_id}] STATE: {status.worker_status.state}"
            elif status.process_alive:
                has_problem = True
                msg = f"[{plugin_id}] ALIVE (status unknown)"
            else:
                msg = f"[{plugin_id}] Terminated"

            if verbose or has_problem:
                logger.info(msg)

    def shutdown(self) -> None:
        """Stop all running plugins and release manager resources (idempotent)."""
        if not self._closed:
            try:
                self.plugin_manager.stop_all_plugins()
            except Exception:
                logger.debug("Error stopping plugins during shutdown", exc_info=True)
            self.close()

    def close(self) -> None:
        """Unload all plugins and release manager resources (idempotent)."""
        if not self._closed:
            self._closed = True
            self.plugin_manager.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
