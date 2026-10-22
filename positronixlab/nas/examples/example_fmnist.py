


from positronixlab.template import Template
from positronixlab.nas.experiment import Experiment


def main():
    template = Template.FromName("fmnist_fc")
    exp = Experiment("test_experiment", template)
    exp.run()


if __name__ == "__main__":
    main()