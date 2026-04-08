
from io import StringIO
import os
import re
import sys
import threading
from modularnet.utils.utils import get_thread_name
import wandb

import warnings


warnings.filterwarnings('ignore', message='.*Full backward hook is firing.*')

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.module')

class Logger():
    LEVELS = ['LOG', 'WARNING', 'ERROR']
    
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_path=None, filename='log', backend='file', show_console=True, show_level=None, capture_threads=False, ignore_warnings=True):
        self.main_thread = threading.current_thread()

        self.log_file = None
        self.backend = backend
        self.base_path = None
        self.filename = filename
        
        self.show_console = show_console
        self.show_level = 'LOG' if show_level is None else show_level
        
        self.files = {} # type: dict[threading.Thread, StringIO]
        self.lock = threading.Lock()
        self.ignore_patterns = [
            re.compile(r'.*AxParameterWarning.*'),
            re.compile(r'.*repr.*attribute.*'),
            re.compile(r'.*Full backward hook.*')
        ]

        
        self.init_backend(backend, base_path, filename)

        if capture_threads: self.capture_stdout()
    
    def init_backend(self, backend=None, base_path=None, filename=None, ):
        self.base_path = base_path if base_path is not None else './logs'
        self.filename = filename
        self.backend = backend
        if base_path is None: return False
        if filename is None: return False
        if backend is None: return False
        
        if backend == 'wandb':
            #TODO: one day
            wandb.init(project=base_path, name=filename)
            wandb.login()
        elif backend == 'file':
            os.makedirs(base_path, exist_ok=True)
            #log_file = f"{base_path}/{filename}.txt"
            #with open(log_file, 'w') as f: f.write('')
        else:
            raise ValueError(f"Unknown backend: {backend}")
        

    
    def console_write(self, text):
        thread_name = get_thread_name()
        if thread_name not in ['main', 'mainthread', 'step_cluster_main','step_ablation_main']: return
        if any(pat.match(text) for pat in self.ignore_patterns): return
        self.old_stdout.write(text)
        
    def set_base_path(self, base_path):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def capture_stdout(self, enable=True):
        if enable:
            if sys.stdout != self:  
                self.old_stdout = sys.stdout
                sys.stdout = self
        else:
            if sys.stdout == self:
                sys.stdout = self.old_stdout
                self.old_stdout = None
    

    def write(self, text):
        thread_name = get_thread_name()
        self.console_write(text)
        # Get or create file for this thread
        if thread_name not in self.files:
            with self.lock:
                if thread_name not in self.files:
                    # Use thread name (which is the target function name) + PID
                    # filename = f"{thread_name}_{os.getpid()}.txt"
                    filename = f"{thread_name}.txt"
                    fullpath = os.path.join(self.base_path, filename)
                    self.old_stdout.write(f"Creating log file for thread {thread_name}: {fullpath}")
                    self.files[thread_name] = open(fullpath, 'w', buffering=1)  # Line buffered
                    self.files[thread_name].write(f"Log file created for thread {filename} ({thread_name}). PID: {os.getpid()}\n\n")
        
        self.files[thread_name].write(text)
        self.files[thread_name].flush()

        return len(text)
    
    def flush(self):
        for f in self.files.values():
            f.flush()
    
    def close_all(self):
        for f in self.files.values():
            f.close()

    def log(self, message, level='LOG'):
        msg = self.format_message(message, level )
        self.print_logs(msg, level)

        if self.backend == 'file':
            self.write_file(msg)
        elif self.backend == 'tensorboard':
            print('TensorBoard logging not implemented yet.')
        elif self.backend == 'wandb':
            wandb.log({level: msg})
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def warning(self, message): self.log(message, 'WARNING')
    def error(self, message): self.log(message, 'ERROR')

    def read_logs(self):
        with open(self.log_file, 'r') as f:
            return f.readlines()
        
    def format_message(self, message, level='LOG'):
        return f'[{level}] {message}'
    
    def write_file(self, message):
        with open(self.log_file, 'a') as f: f.write(message+'\n')
    
    def print_logs(self, msg, level='LOG'):
        if not self.show_console: return
        if Logger.LEVELS.index(level) < Logger.LEVELS.index(self.show_level): return
        print(msg)


if __name__ == '__main__':
    logger = Logger(base_path='logs', filename='test_log', backend='file', show_console=True, show_level='LOG')
    logger.log('This is a log message.')
    logger.warning('This is a warning message.')
    logger.error('This is an error message.')
    
    print("Log file contents:")
    print(''.join(logger.read_logs()))
    logger.log('This is another log message after reading the file.')
    print("Log file contents after writing again:")