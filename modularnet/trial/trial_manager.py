



class TrialManager:
    def __init__(self, trial_cls, *args, **kwargs):
        self.trial_cls = trial_cls
        self.args = args
        self.kwargs = kwargs


    