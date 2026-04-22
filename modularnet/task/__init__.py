"""ModularNet Task System - Parallel execution abstraction.

This module provides a unified interface for executing functions in parallel
contexts (threads or processes), with an abstract base class and concrete
implementations for both threading and multiprocessing.

Classes:
    BaseTask: Abstract base class for parallel tasks
    ThreadTask: Task implementation using threading
    ProcessTask: Task implementation using multiprocessing
    Task: Factory class for creating appropriate task types

Example:
    >>> from modularnet.task import Task
    >>> def my_function(x):
    ...     return x * 2
    >>> task = Task.create(my_function, mode="thread", args=(5,))
    >>> task.start()
    >>> task.join()
"""

from .task import BaseTask, ThreadTask, ProcessTask, Task

__all__ = ['BaseTask', 'ThreadTask', 'ProcessTask', 'Task']
