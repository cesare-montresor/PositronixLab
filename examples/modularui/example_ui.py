import os
import time
from pathlib import Path

from positronixlab.modularui.modularui import ModularUI
from positronixlab.utils import utils


class ExampleUI:
    def __init__(self, workdir):
        self.workdir = workdir

        self.ui = ModularUI()
        self.path_data = Path(self.workdir,"dump","data.pickle")
        self.payload = utils.load_data(self.path_data)  # Load your data here
        
        self.header = self.payload.get("header", {}) if self.payload else {}
        self.data = self.payload.get("data", {}) if self.payload else {}


        self.all_vars = self.data.get("all_vars", {}) if self.data else {}
        self.metrics = self.data.get("metrics", {}) if self.data else {}
        self.stats = self.data.get("stats", {}) if self.data else {}
        self.positions_plot = self.data.get("positions_plot", {}) if self.data else {}
        self.pca_history = self.data.get("pca_history", {}) if self.data else {}

    def main(self):
        print("Starting Example UI.\n",f"Workdir: {self.workdir}\n",f"CWD: {os.getcwd()}\n", f"ui homedir: {self.ui.homedir()}\n")
        self.ui.run()
        self.ui.update_vars(self.all_vars)
    
        while self.ui.is_running():
            try:
                has_data = self._loop()
            except KeyboardInterrupt:
                break  # Allow graceful shutdownw
        self.ui.stop()
    
    def _loop(self):
        has_data = False
        for layer_id, positions in self.positions_plot.items():
            if len(positions) == 0: continue  # Check if there are any positions to pop
            has_data = has_data or True
            next_pos = positions.pop()  # Get the last position if available
            self.ui.add_positions(layer_id, next_pos)

        for layer_id, metric in self.metrics.items():
            for name, values in metric.items():
                if len(values)== 0: continue  # Check if there are any metrics to pop
                has_data = has_data or True
                next_metrics = values.pop()  # Get the last metrics if available
                next_metrics = fix_type(next_metrics)  # Ensure the metrics are in the correct format
                if name == 'spaceloss':
                    self.ui.add_metric_spaceloss(layer_id, next_metrics)
                elif name == 'training':
                    self.ui.add_metric_training(name, next_metrics)

        
        time.sleep(1.0)  # Allow other plugins to access this plugin's state

        return has_data




# fixed in the originsl code, but this data comes from a dump
def fix_type(next_metrics):
    if next_metrics is None: return []

    if isinstance(next_metrics, (tuple)): 
        next_metrics = list(next_metrics)
    if isinstance(next_metrics, (list)):
        for i, v in enumerate(next_metrics):
            if isinstance(v, (list, tuple)):
                next_metrics[i] = fix_type(v)
            elif not isinstance(v, (int, float)):
                next_metrics[i] = float(v)
    elif not isinstance(next_metrics, (int, float)):
        next_metrics = float(next_metrics)
    return next_metrics



if __name__ == "__main__": 
    workdir = os.path.dirname(os.path.realpath(__file__))
    app = ExampleUI(workdir)
    app.main()