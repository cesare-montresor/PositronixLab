"""
Concise CPU example for `positronixlab.template`:
- Loading and customizing a dataset template via `TemplateFactory`
- Defining derived parameters and linear constraints on `TemplateBase`
- Creating & validating the standard modularity ablation variants
- Configuring an Ax 1.3 `Client`, generating a trial, and merging `fixed` + sampled params
- Round-tripping a template to/from YAML
"""

from __future__ import annotations

import os
import tempfile

from ax.api.client import Client

from positronixlab.template import TemplateBase, TemplateFactory
from positronixlab.template.template_base import ParameterType
from positronixlab.template.template_factory import TemplateVariantFactory


def main() -> None:
    # 1. Load a built-in template from TemplateFactory
    print("Available factory templates:", list(TemplateFactory.EXPERIMENTS.keys()))
    tpl = TemplateFactory.FromName("mnist_fc", name="mnist_fc_demo")

    # 2. Customize with a derived parameter, a linear constraint, and a fixed override
    tpl.add_fixed("train_epochs", 5)
    tpl.add_derived("total_hidden_dim", "input_hidden_dim + output_hidden_dim", ParameterType.INT)
    tpl.add_constraint("output_hidden_dim <= input_hidden_dim")
    print(f"Loaded '{tpl.name}': {len(tpl.params)} tunable, {len(tpl.fixed)} fixed, {len(tpl.constraints)} constraint(s)")

    # 3. Create and validate the canonical 4-variant modularity matrix
    variants = TemplateVariantFactory.create_standard_modularity_variants(base_name="mod")
    variants.validate_all(tpl)
    print("Validated variants:", variants.list_variants())

    # 4. Configure an Ax 1.3 Client and sample a trial
    client = Client(random_seed=42)
    client.configure_experiment(
        name=tpl.name,
        parameters=tpl.to_parameter_configs(),
        parameter_constraints=tpl.get_constraints() or None,
    )
    client.configure_optimization(objective="accuracy")

    trials = client.get_next_trials(max_trials=1)
    trial_index, sampled_params = next(iter(trials.items()))
    merged_config = {**tpl.get_fixed(), **sampled_params}
    print(
        f"Sampled trial {trial_index}: "
        f"dataset={merged_config['dataset_name']}, "
        f"lr={merged_config['learning_rate']:.5f}, "
        f"in_dim={merged_config['input_hidden_dim']}, "
        f"out_dim={merged_config['output_hidden_dim']}, "
        f"total_dim={merged_config['total_hidden_dim']}"
    )

    # 5. Save and reload YAML
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, f"{tpl.name}.yaml")
        tpl.save(yaml_path)
        reloaded = TemplateBase.load(yaml_path)
        assert reloaded.to_dict() == tpl.to_dict()
        print(f"YAML round-trip succeeded ({yaml_path})")


if __name__ == "__main__":
    main()
