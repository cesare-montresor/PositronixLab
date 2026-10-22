import os

from ax.adapter import Generators
from ax.analysis.analysis import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationStep
from ax.service.ax_client import AxClient, ObjectiveProperties



from positronixlab.template import Template, TemplateBase

from positronixlab.utils.utils import load_data, save_data




class ExperimentPhase:
    
    @classmethod
    def from_path(cls, path: str) -> 'ExperimentPhase':
        phase = cls()
        phase.load( path )
        return phase

    def __init__(self, name:str=None, client:AxClient=None, template:TemplateBase=None, base_path:str=None, import_phase:str=None):
        self.name = name
        self.client = client
        self.template = template
        self.results = []
        self.times = []
        self.import_phase = import_phase
        self.base_path = base_path  
    
    def add_results(self, results, times):
        self.results.append(results)
        self.times.append(times)
    
    def to_dict(self):
        return {
            'name': self.name,
            'client': self.client.to_json_snapshot(),
            'template': self.template.to_dict(),
            'results': self.results,
            'times': self.times,
            'import_phase': self.import_phase,
            'base_path': self.base_path,
        }
    
    def from_dict(self, data):
        if data is None: return False
        self.name = data.get('name')
        self.client = AxClient.from_json_snapshot( data.get('client') )
        self.template = Template.from_dict( data.get('template') )
        self.results = data.get('results', [])
        self.times = data.get('times', [])
        self.import_phase = data.get('import_phase')
        self.base_path = data.get('base_path')
        return True
    
    def paths(self, path:str=None, makedirs:bool=True):
        if path is None: path = self.base_path
        variant = None
        if self.template is not None:
            variant = self.template.variant
            if variant is None: variant = 'common'
        
        if variant is None:
            variant = os.path.basename(path)

        path_snapshot = os.path.join( path, f'ax_client_snapshot_{variant}.json' )
        path_template = os.path.join( path, f'template_{variant}.yaml' )
        path_phase = os.path.join( path, f'experiment_phase_{variant}.json' )
        path_pareto = os.path.join( path, f'pareto_optimal_results_{variant}.json')
        
        if makedirs:
            os.makedirs( path, exist_ok=True)

        return path_snapshot, path_template, path_phase, path_pareto

    def save(self, path:str=None):
        path_snapshot, path_template, path_phase, path_pareto = self.paths( path )
        
        # save ax client
        self.client.save_to_json_file( path_snapshot )
        # save template
        self.template.save( path_template )
        # save experiment phase
        dump = self.to_dict()
        del dump['client']
        del dump['template']
        
        save_data(path_phase, dump)

        best_results = {}
        try:
            best_results = self.client.get_pareto_optimal_parameters()
            print(f"Found {len(best_results)} pareto optimal results for phase {self.name}")
        except Exception as e:
            print(f"No pareto result available yet for phase {self.name}")
        save_data(path_pareto, best_results)
        
    
    def load(self, path:str=None):
        path_snapshot, path_template, path_phase, _ = self.paths( path )
        
        # load ax client
        self.client = AxClient.load_from_json_file( path_snapshot )

        # load template
        self.template = TemplateBase.load( path_template )

        # load experiment phase
        data = load_data( path_phase )
        self.name = data.get('name')
        self.results = data.get('results', [])
        self.times = data.get('times', [])
        self.import_phase = data.get('import_phase')
        self.base_path = data.get('base_path')
        return True