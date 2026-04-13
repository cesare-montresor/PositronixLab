
from threading import Thread

from torch.multiprocessing import Process, Queue, Event
from typing import Literal, Optional, Tuple, Dict
import os
import time

from abc import ABC, abstractmethod

from modularnet.utils.logger import Logger



class BaseTask(ABC):
    EXEC_THREAD = 'thread'
    EXEC_PROCESS = 'process'
    EXEC_SYNC = 'sync'
    EXEC_LIST = [EXEC_THREAD, EXEC_PROCESS, EXEC_SYNC]
    EXEC_DEFAULT = EXEC_PROCESS

    def __init__(self, exec_type=EXEC_DEFAULT, repeat: int = -1, fatal_errros: bool = False,  autostart: bool = True):
        self.exec_type = exec_type
        if self.exec_type not in self.EXEC_LIST:
            raise ValueError(f"Invalid exec_type '{self.exec_type}'. Must be one of {self.EXEC_LIST}.")
        
        
        self.process = Process(target=self.run)
        self.thread = Thread(target=self.run)
        
        self.in_queue = Queue()
        self.out_queue = Queue()
        self.start_event = Event()
        self.stop_event = Event()
        self.fatal_errors = fatal_errros
        self.repeat = repeat
        

        if autostart:
            self.start()

        
    def start(self, wait: bool = False):
        self.start_event.clear()
        self.stop_event.clear()
        self.process.start()
        if wait:
            self.start_event.wait()
        self.onStart()   
        
    

    def stop(self, wait: bool = True):
        self.stop_event.set()
        self.process.terminate()
        if wait: 
            self.process.join()

    def run(self):
        self.start_event.set()
        Logger().capture_stdout()
        while not self.stop_event.is_set():
            try:
                self.onUpdate()
            except Exception as e:
                print(f"{self.__class__.__name__} encountered an error: {e}")
                import traceback
                traceback.print_exc()
                if self.fatal_errors:
                    self.stop_event.set()
                    raise e


        self.stop_event.clear()
        self.start_event.clear()
        
    def add_data(self, data):
        uid = f"uid_{self.__class__.__name__}_{os.getpid()}_{time.time()}"
        self.in_queue.put( (uid, data) )
        return uid

    def get_result(self, block: bool = True, timeout: Optional[float] = None):
        return self.out_queue.get(block=block, timeout=timeout)
    
    def queue_status(self):
        return self.in_queue.qsize(), self.out_queue.qsize()


    # to override
    def onStart(self): pass
    def onUpdate(self): pass