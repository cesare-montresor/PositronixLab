from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import os
from typing import Any, Callable, Dict, List, Tuple
import parallelazy


def _eval_worker(eval_fn: Callable, trial_index: int, parameters: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
    metrics = eval_fn(parameters, trial_index)
    return trial_index, metrics


def run_trials_parallel(
    trial_batch: Dict[int, Dict[str, Any]],
    eval_fn: Callable[[Dict[str, Any], int], Dict[str, float]],
    max_workers: int = 4,
) -> List[Tuple[int, Dict[str, float]]]:
    """Runs a batch of trials in parallel across worker processes."""
    if not trial_batch:
        return []

    if max_workers <= 1 or len(trial_batch) == 1:
        return [(idx, eval_fn(params, idx)) for idx, params in trial_batch.items()]

    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(trial_batch)), mp_context=ctx) as executor:
        futures = {
            executor.submit(_eval_worker, eval_fn, idx, params): idx
            for idx, params in trial_batch.items()
        }
        for future in as_completed(futures):
            results.append(future.result())

    return results


def run_variants_parallel(
    variant_tasks: List[Tuple[str, Callable[[], Any]]],
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Runs multiple variant optimization loops concurrently in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(variant_tasks))) as executor:
        futures = {
            executor.submit(fn): name
            for name, fn in variant_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
    return results
