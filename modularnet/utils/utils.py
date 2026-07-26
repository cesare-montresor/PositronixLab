
import contextlib
import logging
import pickle
import threading
import time
import warnings
import numpy as np
import re
import torch
import os
import json
import yaml
from hashlib import md5
import multiprocessing as mp

debug = False

last_session_path = None


logger = logging.getLogger(__name__)



def init_debugger(port: int, process_name: str = 'worker', wait_for_client: bool = False) -> None:
    try:
        import debugpy
        debugpy.listen(("localhost", port))
        logger.info(f"Debugger listening on port {port}")  # extension sees this
        
        print(f"Debugger listening on port {port} for {process_name} process", flush=True)  # belt and braces
        if wait_for_client:
            logger.info("Waiting for debugger to attach...")
            debugpy.wait_for_client()        # blocks AFTER the log line is out
            logger.info("Debugger attached")
    except Exception as e:
        logger.warning(f"Failed to initialize debugger on port {port}: {e}")


def process_start_spawn(force=False):
    current = mp.get_start_method(allow_none=True)
    if current not in ('spawn', None):
        warnings.warn(f"Expected spawn start method, got '{current}'. CUDA may be unreliable.")
    elif current is None:
        mp.set_start_method('spawn', force=force)


@contextlib.contextmanager
def suppress_output(stderr: bool = True):
    stderr_fd = 2 if stderr else 1  # C-level stderr
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(devnull)
        os.close(old_stderr)



def session_path(name, uid=None, createDirectory = True, force=False):   
    global last_session_path 
    if last_session_path is not None and not force: return last_session_path
    
    basedir = 'output/'
    uid = str(uid) if uid is not None else str(time.time()).replace('.','_')
    path = basedir + name + "_" + uid + "/"
    last_session_path = path
    if createDirectory: os.makedirs(path, exist_ok=True)
    return path



def camel_to_kebab(text):
    # Find transitions from lowercase/number to uppercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', text)
    # Find transitions from lowercase to uppercase/number
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
    return s2.lower()

def camel_to_snake(text):
    # Find transitions from lowercase/number to uppercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    # Find transitions from lowercase to uppercase/number
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def get_thread_name():
    thread = threading.current_thread()

    thread_name = thread.name.lower()
    target_name = re.compile(r'\((.+)\)').search(thread_name)
    if target_name is not None: thread_name = target_name.group(1)
    
    thread_name = thread_name.replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_')

    return thread_name

def assert_tensor(tensor:torch.Tensor, name="tensor"):
    valid = is_valid_tensor(tensor)
    if valid is not True:
        import traceback
        raise ValueError(f"Invalid tensor: {name}, reason: {valid}", tensor, traceback.format_stack())

def is_valid_tensor(tensor:torch.Tensor):
    if tensor is None: return "none"
    if not isinstance(tensor, torch.Tensor): return "not_tensor"
    if tensor.numel() == 0: return "empty"
    if torch.isnan(tensor).any().item(): return "nan"
    if torch.isinf(tensor).any().item(): return "inf"
    return True


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def check_tensor(tensor, name="tensor", debug = debug):
    if not debug: return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN detected in {name}")
        info_tensor(tensor, name)
        raise Exception("NaN detected")
    
def info_tensor(tensor, name="tensor", threshold=1e-10, debug = debug):
    if not debug: return
    print(f"-----")
    print(f"Tensor: {name}")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min()}")
    print(f"Max: {tensor.max()}")
    print(f"Mean: {tensor.mean()}")
    if torch.isnan(tensor).any().item(): print("NaNs!")
    if torch.isinf(tensor).any().item(): print("Infs!")
    near_zero_count = (torch.abs(tensor) < threshold).sum()
    if near_zero_count>0: print(f"Zeros: {near_zero_count}")
    print(f"=====")

def module_name(module):
    return module.__class__.__module__

def class_name(module):
    return module.__class__.__name__


def truth_table_from_cm(confusion_matrix:np.ndarray):
    truth_table = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i] # True Positives: diagonal element
        fp = np.sum(confusion_matrix[:, i]) - tp # False Positives: sum of column i, excluding diagonal
        fn = np.sum(confusion_matrix[i, :]) - tp # False Negatives: sum of row i, excluding diagonal  
        tn = np.sum(confusion_matrix) - tp - fp - fn # True Negatives: total - tp - fp - fn
        truth_table.append(np.array((tp, fp, fn, tn)))
    return  np.array(truth_table)

def f1s_from_truth_table(tt:np.ndarray):
    f1s = []
    for tp, fp, fn, tn in tt:
        f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0
        f1s.append(f1)
    return np.array(f1s)


def set_best(bests, key, val, min=True):
    # 1. Handle new keys
    if key not in bests:
        bests[key] = val
        return 1, None
    
    # 2. Check for improvement
    if (min and val < bests[key]) or (not min and val > bests[key]):
        
        # Prevent ZeroDivisionError if the old best was exactly 0
        if bests[key] == 0:
            diff = 1.0
        else:
            diff = abs((bests[key] - val) / bests[key])
            
        old_val = bests[key]
        bests[key] = val
        return diff, old_val

    # 3. No improvement
    return -1, bests[key]

def signature_settings(data:dict) -> str:
    content  = json.dumps(data, sort_keys=True)
    content = content.encode()
    sign = md5(content).hexdigest()
    #print(f"Settings signature: {sign}\n\n---\n\n{data}\n\n---\n\n")
    return sign



def save_data(path:str, data):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    if ext == '.json':
        with open(path, 'w') as f: 
            json.dump(data, f, indent=2)
    elif ext == '.yaml':
        with open(path, 'w') as f: 
            yaml.dump(data, f)
    elif ext in ['.pkl', '.pickle']:
        with open(path, 'wb') as f: 
            pickle.dump(data, f)
    else:
        logging.error(f"Unsupported extension: {ext}")
        return False
    return True


def load_data(path:str):
    path = str(path)
    if not os.path.exists(path): return None
    ext = os.path.splitext(path)[1].strip()
    
    if ext == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif ext == '.yaml':
        with open(path, 'r') as f:
            return yaml.load(f)
    elif ext in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        logging.error(f"Unsupported extension: {ext}")
    return None


def is_debugpy_attached() -> bool:
    import debugpy
    return debugpy.is_client_connected()