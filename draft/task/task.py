import threading
import multiprocessing
from abc import ABC, abstractmethod
from typing import Callable, Any

# Set multiprocessing start method to fork for compatibility with Events
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set or not available


class BaseTask(ABC):
    """Abstract base class for parallel tasks."""
    
    @abstractmethod
    def start(self) -> None:
        """Start the task."""
        pass
    
    @abstractmethod
    def join(self) -> None:
        """Wait for the task to complete."""
        pass
    
    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the task is still running."""
        pass
    


class ThreadTask(BaseTask):
    """Task that runs in a separate thread."""
    
    def __init__(self, target: Callable, args: tuple = (), kwargs: dict = None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.thread = threading.Thread(target=target, args=self.args, kwargs=self.kwargs)
    
    def start(self) -> None:
        """Start the thread."""
        self.thread.start()
    
    def join(self) -> None:
        """Wait for the thread to complete."""
        self.thread.join()
    
    def is_alive(self) -> bool:
        """Check if the thread is still running."""
        return self.thread.is_alive()


class ProcessTask(BaseTask):
    """Task that runs in a separate process."""
    
    def __init__(self, target: Callable, args: tuple = (), kwargs: dict = None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.process = multiprocessing.Process(target=target, args=self.args, kwargs=self.kwargs)
    
    def start(self) -> None:
        """Start the process."""
        self.process.start()
    
    def join(self) -> None:
        """Wait for the process to complete."""
        self.process.join()
        
    
    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process.is_alive()


class Task:
    """Factory class to create parallel tasks (thread or process)."""
    
    @staticmethod
    def create(target: Callable, mode: str = "thread", args: tuple = (), kwargs: dict = None) -> BaseTask:
        """
        Create a parallel task.
        
        Args:
            target: The function to execute.
            mode: "thread" for threading or "process" for multiprocessing.
            args: Positional arguments for the target function.
            kwargs: Keyword arguments for the target function.
        
        Returns:
            A BaseTask instance (ThreadTask or ProcessTask).
        
        Raises:
            ValueError: If mode is not "thread" or "process".
        """
        if mode == "thread":
            return ThreadTask(target, args, kwargs)
        elif mode == "process":
            return ProcessTask(target, args, kwargs)
        else:
            raise ValueError(f"Mode must be 'thread' or 'process', got '{mode}'")