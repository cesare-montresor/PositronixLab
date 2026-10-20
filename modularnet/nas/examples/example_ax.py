"""Run a template-driven Ax experiment with a deterministic fake evaluator."""

from pathlib import Path
from modularnet.nas import AxExperiment
from modularnet.template.template_base import TemplateBase


def evaluate(config, metadata):
    del metadata
    width = float(config["input_hidden_dim"])
    dropout = float(config["dropout"])
    return {"accuracy": 1.0 - abs(width - 256) / 256.0 - dropout * 0.1}


if __name__ == "__main__":
    experiment = AxExperiment(
        TemplateBase("example"), 
        Path("ax-example.sqlite"),
        {"accuracy": "maximize"}, 
        name="template_example",
        evaluator=evaluate, 
        max_parallelism=4,
    )
    result = experiment.run(
        trials=4,
        variants={"a": {"exp_metric_modularity": False},
                  "b": {"exp_metric_modularity": True}},
    )
    print(result.best_trials)
