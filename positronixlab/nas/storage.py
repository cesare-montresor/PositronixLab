import os
from typing import Any, List, Optional, Sequence, Union

from ax.api.client import Client
from ax.api.configs import (
    RangeParameterConfig,
    ChoiceParameterConfig,
    StorageConfig,
)


def get_storage_config(db_path: str) -> StorageConfig:
    """Creates a StorageConfig pointing to the specified SQLite database file."""
    abs_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    return StorageConfig(url=f"sqlite:///{abs_path}")


def get_or_create_client(
    experiment_name: str,
    db_path: str,
    parameters: Optional[List[Union[RangeParameterConfig, ChoiceParameterConfig]]] = None,
    objectives: Optional[str] = None,
    outcome_constraints: Optional[Sequence[str]] = None,
) -> Client:
    """
    Implements experiment resumption following https://ax.dev/docs/recipes/experiment-to-sqlite:
    - If the database file exists and contains the experiment, loads it to resume.
    - Otherwise, initializes a new Client, configures experiment and optimization, and persists to SQLite.
    """
    abs_path = os.path.abspath(db_path)
    storage_config = get_storage_config(abs_path)

    if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
        try:
            client = Client.load_from_database(
                experiment_name=experiment_name,
                storage_config=storage_config,
            )
            return client
        except Exception:
            # If load fails (e.g. empty or incomplete schema), initialize fresh
            pass

    client = Client(storage_config=storage_config)

    if parameters is not None:
        client.configure_experiment(
            name=experiment_name,
            parameters=parameters,
        )

    if objectives is not None:
        client.configure_optimization(
            objective=objectives,
            outcome_constraints=outcome_constraints,
        )

    return client
