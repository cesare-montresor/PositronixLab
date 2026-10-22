
import glob
import json
import os
import threading

import traceback
import time

import logging

import torch


from ax.adapter import Generators
from ax.analysis.analysis import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationStep
from ax.service.ax_client import AxClient, ObjectiveProperties


from positronixlab.template import TemplateBase
from positronixlab.utils.utils import load_data, save_data, signature_settings, process_start_spawn


from positronixlab.nas.phase import ExperimentPhase

logger = logging.getLogger(__name__)


process_start_spawn(force=True) 




    

class Experiment:
    OUTPUT_DIR = 'output/'
    TRIAL_DIR = 'trials/'
    CONFIG_DIR = 'config/'
    PHASE_DIR = 'phase/'
    LOG_DIR = 'logs/'
    REPORT_DIR = 'reports/'
    
    @classmethod
    def list_experiments(cls, path=OUTPUT_DIR):
        paths = glob.glob( f'{path}*' )
        valid_paths = []
        for path in paths:
            if not os.path.isdir( path ): continue
            exp_path =  cls.get_experiment_path(path)
            if exp_path is None: continue
            valid_paths.append( path )
        return valid_paths
    
    @classmethod
    def from_path(cls, path):
        config_path = cls.get_experiment_path(path)
        if config_path is None:
            print(f"Error: experiment folder {path} has no experiment file:")
            return None
        
        with open( config_path ) as cfg: config = json.load( cfg )
        if config is None:
            print(f"Error: config is not a valid JSON file {config_path}")
            return None
        
        exp = Experiment(config=config)
        if exp.load():
            print(f"Success: experiment {exp.name}")
        else:
            print(f"Error: experiment could not be loaded {path} ")
            return None

        return exp
    
    @classmethod
    def get_experiment_path(cls, path):
        config_path = os.path.join(path, 'config', 'experiment.json')
        if not os.path.exists( config_path ):
            #print(f"Error: experiment folder {path} has no experiment file: {config_path}")
            return None
        return config_path
    
    def worker_num(self, num):
        num = min( num, os.cpu_count()-1 )
        num = max( 1, num )
        return num


    def __init__(self, 
            config:dict=None, 
            name:str=None, 
            experiment:str= None, 
            variants:list[str]=None, 
            seed = None, 
            debug=False,
            trials_prior=30, 
            trials_search=30, 
            parallel_trials=None,
            draft=False, 
            device='cuda', 
            save_video_space=False,
            save_history=False,
            num_workers_data=None,
        ):
        
        if config is not None:
            self.from_dict( config )
        else:
            self.experiment = experiment if experiment is not None else MetaModelTemplate.EXPERIMENTS[0]
            self.variants = variants if variants is not None else MetaModelTemplate.VARIANTS
            self.name = name if name is not None else f"TEST_RUN_{int(time.time())}"

            #self.base_path = session_path(name, uid=self.signature, force=True)

            #self.templates = [ MetaModelTemplate.get_experiment_variant( self.experiment, var ) for var in self.variants ]
            
            self.seed = seed if seed is not None else int(time.time()) % (2**32 - 1) # random seed by default
            
            
            
            self.debug = debug
            self.trials_prior = trials_prior
            self.trials_search = trials_search

            self.parallel_trials = parallel_trials if parallel_trials is not None else len(self.variants)
            
            self.draft = draft
            self.device = device
            self.save_video_space = save_video_space
            self.save_history = save_history
            self.num_workers_data = 0 if num_workers_data is None else num_workers_data
            
            #self.num_workers_data = self.worker_num( self.num_workers_data )
            self.parallel_trials = self.worker_num( self.parallel_trials )
            
            self.phases = {} #type: dict[str, ExperimentPhase]

        threading.current_thread().name = f"exp_{self.name}"
        self.logger = Logger()

        if self.load():
            print(f"Loaded existing experiment {self.name} from {self.path_home()}")
        else:
            print(f"Initialized new experiment {self.name} at {self.path_home()}")
            self.init_phases()


        self.save()
        
    def add_phase(self, name:str, client:AxClient, template:TemplateBase, import_phase:str=None):
        phase_path = self.path_phase(name)
        phase = ExperimentPhase( name, client, template, base_path=phase_path, import_phase=import_phase )
        self.phases[name] = phase
        phase.save()
        return phase    
        
    def signature(self):
        fixed_fields = self.to_dict()
        #fixed_fields['template'] = self.template.signature()
        sign = signature_settings(fixed_fields)
        print(f"Experiment signature: {sign}\n{fixed_fields}")
        return sign


    def to_dict(self):
        return {
            'name': self.name,
            'experiment': self.experiment,
            'variants': self.variants,
            'seed': self.seed,
            'trials_prior': self.trials_prior,
            'trials_search': self.trials_search,
            'parallel_trials': self.parallel_trials,
            'draft': self.draft,
            'device': self.device,
            'save_video_space': self.save_video_space,
            'save_history': self.save_history,
            'num_workers_data': self.num_workers_data,
            'phases': list(self.phases.keys()),
        }


    def from_dict(self, data):
        if data is None: return False
        self.name = data['name']
        self.seed = data['seed']
        self.trials_prior = data['trials_prior']
        self.trials_search = data['trials_search']
        self.parallel_trials = data['parallel_trials']
        self.draft = data['draft']
        self.device = data['device']
        self.save_video_space = data['save_video_space']
        self.save_history = data['save_history']
        self.num_workers_data = data['num_workers_data']
        self.experiment = data['experiment']
        self.variants = data['variants']

        self.phases = {}
        
        for name in data['phases']:
            path = self.path_phase( name, makedirs=False )
            self.phases[name] = ExperimentPhase.from_path( path )

        return True


    def makedirs(self, path:str):
        path = path.strip()

        if os.path.isfile(path):
            path = os.path.dirname(path)
        elif not os.path.exists(path) and not path.endswith('/'):
            path = os.path.dirname(path)
            
        os.makedirs( path, exist_ok=True)

    def path_home(self,rel_path=None, makedirs=True):
        exp_ui = self.name + "_" + self.experiment + "_" + "".join(self.variants) #self.signature() # cache by template ( any change => new template => new signature => new experiment )
        path = os.path.join(Experiment.OUTPUT_DIR, exp_ui )
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path

    def path_trial(self, variant, trial_num, rel_path=None, makedirs=True):
        if variant is None: variant = 'common'    
        path = self.path_phase( os.path.join(variant, Experiment.TRIAL_DIR, f't{trial_num}') )
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path
    
    def path_phase(self, rel_path=None, makedirs=True):
        path = self.path_home(Experiment.PHASE_DIR)
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path

    def path_config(self,rel_path=None, makedirs=True):
        path = self.path_home(Experiment.CONFIG_DIR)
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path
    
    def path_logs(self,rel_path=None, makedirs=True):
        path = self.path_home(Experiment.LOG_DIR)
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path
    
    def path_report(self,rel_path=None, makedirs=True):
        path = self.path_home(Experiment.REPORT_DIR)
        if rel_path is not None: path = os.path.join(path, rel_path)
        if makedirs: self.makedirs(path)
        return path

    def path_snapshot(self, variant:str=None):
        if variant is None: variant = 'common'
        return self.path_config(f'ax_client_snapshot_{variant}.json')

    def path_template(self):
        return self.path_config('template.yaml')  
    
    def path_experiment(self):
        return self.path_config('experiment.json')  
    
    
    def save(self):
        
        save_data(  self.path_experiment(), self.to_dict() )

    def load(self):
        data = load_data( self.path_experiment() )
        
        return self.from_dict( data )


    def configure_client(self, template:TemplateBase, strategy:str='bo'):
        # seed_everything(self.seed) -> seed what ? ax client ?
        params = template.get_params()

        min_acc = template.get_value('nas_min_acc', 0.8) 
        objectives = {'val_acc': ObjectiveProperties(minimize=False, threshold=min_acc)}

        mod_metric = template.get_value('exp_metric_modularity', [False])
        if any(mod_metric):
            min_mod_score = template.get_value('nas_min_mod_score', 0.2)
            objectives['modularity_score'] = ObjectiveProperties(minimize=False, threshold=min_mod_score) 
        
        if strategy == 'sobol':
            generation_steps = [ GenerationStep(Generators.SOBOL, self.trials_prior) ]
        else:
            generation_steps = [ GenerationStep(Generators.BOTORCH_MODULAR, self.trials_search) ]
        gs = GenerationStrategy(steps=generation_steps)
        

        ax_client = AxClient(generation_strategy=gs)
        ax_client.create_experiment( 
            name=self.name, 
            parameters=params,
            objectives=objectives,
            tracking_metric_names=ModularStep.METRICS
        )

        
        
        return ax_client


    def init_phases(self):
        self.logger = Logger()
        self.logger.init_backend(backend='file', base_path=self.path_logs(), filename='log')
        self.logger.capture_stdout()

        
        common_template = MetaModelTemplate.get_experiment( self.experiment ).get_variant()
        # prepare common phase - random

        common_client = self.configure_client(common_template, strategy='sobol') 
        self.add_phase( 'common', common_client, common_template )
        
        # prepare variants phase - search
        for variant in self.variants:
            va_template = common_template.get_variant( variant )
            va_client = self.configure_client( va_template, strategy='bo')
            phase = self.add_phase( variant, va_client, va_template, import_phase='common' )



    def run(self, blocking=True):
        # run phase: common -> sequential phase -> parallel random trials
        common_phase = self.phases.get('common')
        self.run_phase(common_phase, parallel_trials=self.parallel_trials)
        
        # run phases: variants -> parallel phases -> sequential search trials 
        phase_threads = []
        for variant in self.variants:
            variant_phase = self.phases.get(variant)
            self.run_phase( variant_phase, 1 )
            phase_thread = threading.Thread(target=self.run_phase, args=( variant_phase, 1 ))
            phase_thread.name =  f"exp_{self.name}_phase_{variant}"
            phase_thread.start()
            phase_threads.append( phase_thread )
        
        if blocking:
            for phase_thread in phase_threads:
                phase_thread.join()


    def run_phase(self, phase: ExperimentPhase, parallel_trials:int=1):
        times = []
        stop = False
        num_trials = phase.client.experiment.num_trials
        variant = phase.template.variant
        if variant is None: variant = 'common'
        
        old_phase = self.phases.get( phase.import_phase, None )
        if old_phase is not None and phase.client.experiment.num_trials == 0:
            phase.client.experiment.warm_start_from_old_experiment( 
                old_phase.client.experiment,
                search_space_check_membership_raise_error=False
            )
            #Experiment.import_trials( old_phase.client, phase.client )
            #phase.client.generation_strategy._maybe_transition_to_next_node(False)

            #phase.client.fit_model()

        options = {
            'name': self.name,
            'variant': variant,
            'debug': self.debug,
            'draft': self.draft,
            'device': self.device,
            'save_video_space': self.save_video_space,
            'seed': self.seed,
            'save_model': False,
            'num_workers_data': self.num_workers_data,
            'save_history': self.save_history,  
            'home_path': self.path_home(),
        }

        
        while not stop:
            trials, stop = phase.client.get_next_trials(max_trials=parallel_trials)
            results_queue = mp.Queue()
            result_cnt = 0 
            trials_procs = []
            for trial_index, parameters in trials.items():
                opts = options.copy()
                opts['trial_path'] = self.path_trial( variant, trial_index )
                args = (trial_index, parameters, opts, results_queue) 
                proc = mp.Process(None, Experiment.run_trial, args=args )
                proc.name = f"exp_{self.name}_{variant}_trial_{trial_index}"
                proc.start()
                trials_procs.append( proc )
                result_cnt += 1
            
            results = []
            while result_cnt > 0:
                result = results_queue.get()
                results.append( result )
                result_cnt -= 1

            for proc in trials_procs:
                proc.kill()
                proc.join()
            
            #results = list(self.trialsPool.starmap( Experiment.run_trial, trials_params))
            
            for trial_index, parameters, metrics, elapsed_time in results:
                # keep only watched metrics 
                metrics = {key:metrics[key][-1] for key in metrics if len(metrics[key])>0 } 
                
                print(metrics)

                phase.client.complete_trial(trial_index=trial_index, raw_data=metrics)
                phase.add_results(metrics, elapsed_time)
                
                #self.save_results()
            
                times.append(elapsed_time)
                avg_time = sum(times) / len(times)
                est_remaining = avg_time * (num_trials - (trial_index + 1))
                print(f"Trial {trial_index} completed in {elapsed_time:.2f} seconds. Estimated remaining time: {est_remaining/60:.2f} minutes.")
                phase.save()
    #def run_trial(self, variant, trial_index, parameters):
    #    pass

    @staticmethod
    def run_trial(trial_index:int, parameters:dict, options:dict, results:mp.Queue):
        elapsed_time = time.time()
        metric = None
        
        while True:
            #print("#"*80)
            print(f"Trial {trial_index} with parameters:")
            #print("#"*80)
            #print(json.dumps(parameters, indent=2))

            try:
                metric = Experiment.train_evaluate(trial_index, parameters, options)
                break
            except torch.cuda.OutOfMemoryError as e:
                print(f"[ERROR] CUDA Out of Memory during trial, retry {trial_index}: {e}")
                print(traceback.format_exc())
                time.sleep(10)
                # retry until OOM is resolved
            except Exception as e:
                print(f"[ERROR] Exception during trial {trial_index}: {e}")
                print(traceback.format_exc())
                raise e
                

            
        elapsed_time = time.time() - elapsed_time
        result = (trial_index, parameters, metric, elapsed_time)
        if results is not None:
            results.put( result )

        return result

    @staticmethod
    def import_trials(from_client:AxClient, to_client:AxClient):
        imported_trials=[]
        for trial in from_client.experiment.trials.values():
            if not trial.status.is_completed: continue
            imported_trials.append( trial )
            trial_data = trial.fetch_data()
            
            params = trial.arms[0].parameters
            results = trial_data.df[["metric_name", "mean"]].to_dict(orient='RECORDS')
            results = {  item['metric_name']: item['mean'] for item in results }
            
            params, trial_num = to_client.attach_trial(params)
            to_client.complete_trial( trial_index=trial_num, raw_data= results )
            print("STEP:" + str(to_client.generation_strategy.current_step_index))
                                      
        #imported_trials = to_client.experiment.warm_start_from_old_experiment(from_client.experiment, search_space_check_membership_raise_error=False)
        print(f"Imported {len(imported_trials)}/{len(from_client.experiment.trials)} trials from previous experiment")


    @staticmethod
    def train_evaluate(trial_index:int, config:dict, options:dict ):
        

        config = MetaModelConfig(config)
        name = options.get('name')
        variant = options.get('variant','')
        trial_path = options.get('trial_path')
        home_path = options.get('home_path')
        debug = options.get('debug', False)
        draft = options.get('draft', False)
        device = options.get('device', 'cuda')
        save_video_space = options.get('save_video_space', False)
        seed = options.get('seed', None)
        save_model = options.get('save_model', False)
        num_workers_data = options.get('num_workers_data', 0)
        save_history = options.get('save_history', False)

        # Code to load the experiment
        logger = Logger()
        logger.init_backend(backend='file', base_path=trial_path, filename='log')
        logger.capture_stdout()

        #seed_everything(seed)
        name = f'{name}_v{variant}_t{trial_index}'
        
        # Create model first to determine device
        model = MetaModel(name, config, device=device)

        if debug: model.print()

        device  = model.device
        # Pass model device to dataset
        dataset = DataProvider.get_dataprovider(config, device=device, num_workers=num_workers_data)
        
        space_size = config.modular_space_size
        # Pass model device to space
        space = ModularSpace(config, name, model, space_size, debug=debug, useActivations=True, useLayers=True, base_path=trial_path, device=device, history=save_history )
        
        steps = ModularStep(config, name, trial_path, dataset, model, space, draft=draft)


        steps.space.modularizer = config.exp_modularizer
        
                
        steps.step_run( save_video=save_video_space)

        if save_model:
            model.save(path=os.path.join( trial_path, 'model_final.pth' ))

        return steps.metrics



if __name__ == "__main__": test()






