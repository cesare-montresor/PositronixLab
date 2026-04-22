# Task System Documentation

## Overview

The Task system provides a unified interface for executing functions in parallel contexts (threads or processes). It abstracts the complexity of choosing between threading and multiprocessing, offering a simple, consistent API for both execution models.

## Architecture

### Class Hierarchy

```
BaseTask (ABC)
├── ThreadTask
└── ProcessTask

Task (Factory)
```

### Design Patterns

- **Abstract Base Class**: `BaseTask` defines the contract all task implementations must follow
- **Factory Pattern**: `Task.create()` encapsulates task creation logic
- **Process/Thread Isolation**: Each task runs independently, minimizing interference

## Core Components

### BaseTask

Abstract base class that defines the interface for parallel execution.

**Purpose**: Establish a contract for any task implementation, ensuring consistent behavior across threading and multiprocessing contexts.

**Abstract Methods**:

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `run()` | `run()` | `Any` | Execute the task logic (must be implemented by subclasses) |
| `start()` | `start()` | `None` | Start the task in its execution context |
| `join()` | `join()` | `None` | Wait for the task to complete |
| `is_alive()` | `is_alive()` | `bool` | Check if the task is currently running |

**Usage Pattern**:
```python
# Do not instantiate directly; use subclasses or Task factory
task = ThreadTask(my_function, args=(arg1, arg2), kwargs={'key': 'value'})
task.start()
task.join()
```

### ThreadTask

Executes a function in a separate thread using Python's `threading` module.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `Callable` | Required | The function to execute |
| `args` | `tuple` | `()` | Positional arguments for the target function |
| `kwargs` | `dict` | `None` | Keyword arguments for the target function |

**Characteristics**:
- Lightweight, shares memory space with parent process
- Suitable for I/O-bound tasks
- Subject to Python's Global Interpreter Lock (GIL)
- Fast creation and startup
- Good for concurrent I/O operations, API calls, file operations

**Example**:
```python
def fetch_data(url):
    response = requests.get(url)
    return response.json()

task = ThreadTask(fetch_data, args=("https://api.example.com/data",))
task.start()
task.join()
print(f"Task alive: {task.is_alive()}")  # False
```

**When to Use**:
- I/O-bound operations (network requests, file I/O, database queries)
- Quick concurrent tasks
- When memory overhead must be minimized
- When sharing memory with parent is beneficial

**Limitations**:
- GIL prevents true parallelism for CPU-bound work
- Not suitable for CPU-intensive tasks
- Shared memory can lead to race conditions

### ProcessTask

Executes a function in a separate process using Python's `multiprocessing` module.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `Callable` | Required | The function to execute |
| `args` | `tuple` | `()` | Positional arguments for the target function |
| `kwargs` | `dict` | `None` | Keyword arguments for the target function |

**Characteristics**:
- True parallelism, separate Python interpreter
- Suitable for CPU-bound tasks
- Heavier memory footprint than threads
- Slower creation and startup
- Process isolation provides safety

**Example**:
```python
def expensive_computation(n):
    result = 0
    for i in range(n):
        result += i ** 2
    return result

task = ProcessTask(expensive_computation, args=(1000000,))
task.start()
task.join()
print(f"Task alive: {task.is_alive()}")  # False
```

**When to Use**:
- CPU-bound operations (data processing, numerical computations)
- Long-running computations
- When process isolation and safety is important
- Machine learning inference or training

**Limitations**:
- Higher memory overhead due to separate interpreter
- Slower startup time
- Data serialization overhead (pickling)
- Cannot directly share mutable objects

### Task Factory

Static factory class that creates appropriate task types based on the requested mode.

**Method**: `Task.create()`

**Signature**:
```python
@staticmethod
def create(
    target: Callable,
    mode: str = "thread",
    args: tuple = (),
    kwargs: dict = None
) -> BaseTask
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `Callable` | Required | The function to execute |
| `mode` | `str` | `"thread"` | Execution mode: `"thread"` or `"process"` |
| `args` | `tuple` | `()` | Positional arguments for the target function |
| `kwargs` | `dict` | `None` | Keyword arguments for the target function |

**Returns**: `BaseTask` instance (either `ThreadTask` or `ProcessTask`)

**Raises**: `ValueError` if mode is not `"thread"` or `"process"`

**Example**:
```python
# Create a thread task
task = Task.create(my_function, mode="thread", args=(1, 2), kwargs={'x': 10})
task.start()

# Create a process task
task = Task.create(my_function, mode="process", args=(1, 2), kwargs={'x': 10})
task.start()
task.join()
```

## Usage Patterns

### Pattern 1: Simple Function Execution

Execute a simple function in a thread:

```python
from modularnet.task import ThreadTask

def greet(name):
    print(f"Hello, {name}!")

task = ThreadTask(greet, args=("Alice",))
task.start()
task.join()
print("Task completed")
```

### Pattern 2: Using the Factory for Mode Selection

Choose execution mode at runtime:

```python
from modularnet.task import Task

def process_data(data):
    # Process large dataset
    return sum(data)

mode = "process" if len(data) > 1_000_000 else "thread"
task = Task.create(process_data, mode=mode, args=(data,))
task.start()
task.join()
```

### Pattern 3: Monitoring Task Status

Check if a task is still running:

```python
from modularnet.task import ThreadTask
import time

def long_task():
    for i in range(10):
        print(f"Step {i}")
        time.sleep(1)

task = ThreadTask(long_task)
task.start()

while task.is_alive():
    print("Task is still running...")
    time.sleep(2)

print("Task finished")
task.join()
```

### Pattern 4: Passing Complex Arguments

Use kwargs for complex parameter passing:

```python
from modularnet.task import ProcessTask

def train_model(model, optimizer, learning_rate=0.001):
    # Training logic
    pass

task = ProcessTask(
    train_model,
    args=(model,),
    kwargs={'optimizer': optimizer, 'learning_rate': 0.01}
)
task.start()
task.join()
```

### Pattern 5: Multiple Concurrent Tasks

Run multiple tasks concurrently:

```python
from modularnet.task import ThreadTask

def fetch_user(user_id):
    # Simulate API call
    import time
    time.sleep(1)
    return f"User {user_id}"

tasks = [ThreadTask(fetch_user, args=(i,)) for i in range(5)]

# Start all tasks
for task in tasks:
    task.start()

# Wait for all to complete
for task in tasks:
    task.join()

print("All tasks completed")
```

## Decision Guide: Thread vs Process

| Aspect | ThreadTask | ProcessTask |
|--------|-----------|-------------|
| **Best For** | I/O-bound work | CPU-bound work |
| **Memory** | Shared, lower overhead | Separate, higher overhead |
| **Startup** | Fast | Slower |
| **GIL Impact** | Subject to GIL | No GIL |
| **Data Sharing** | Direct access | Pickling required |
| **Safety** | Need synchronization | Process isolation |
| **Use Cases** | Web requests, file I/O, DB queries | ML training, data processing, numerical computations |

## Error Handling

### Example: Handling Exceptions in Tasks

```python
from modularnet.task import ThreadTask
import traceback

def risky_operation():
    raise ValueError("Something went wrong")

task = ThreadTask(risky_operation)
task.start()

try:
    task.join()
except Exception as e:
    print(f"Task failed: {e}")
    traceback.print_exc()
```

**Note**: Exceptions in the task target function are not automatically propagated to the parent process/thread. You may need to implement custom exception handling mechanisms if error reporting is critical.

## Best Practices

1. **Choose the Right Mode**:
   - Use `"thread"` for I/O operations
   - Use `"process"` for CPU-intensive work

2. **Always Join Before Exit**:
   ```python
   task.start()
   # ... do other work ...
   task.join()  # Ensure task completes before exiting
   ```

3. **Check Task Status**:
   ```python
   if task.is_alive():
       task.join(timeout=5)  # Wait up to 5 seconds
   ```

4. **Use Factory for Flexibility**:
   ```python
   # Makes it easy to switch modes later
   task = Task.create(fn, mode=mode_from_config)
   ```

5. **Handle Large Data Carefully**:
   - With ProcessTask, large arguments are pickled (slow)
   - Consider using multiprocessing.Queue for large data transfer

6. **Test Both Modes**:
   - Test with ThreadTask for quick iteration
   - Test with ProcessTask for production workloads

## Troubleshooting

### Task Appears to Hang

**Problem**: `task.join()` seems to hang indefinitely

**Solutions**:
- Ensure the target function doesn't have infinite loops
- Add timeout: `task.join(timeout=30)`
- Check if the target function is waiting on I/O that's never satisfied

### Memory Leaks with ProcessTask

**Problem**: Memory usage grows with each ProcessTask created

**Solutions**:
- Ensure tasks are properly joined: `task.join()`
- Process objects hold resources until garbage collected
- Consider using a task pool instead of creating many individual tasks

### Pickling Errors with ProcessTask

**Problem**: `ProcessTask` raises "PicklingError"

**Solutions**:
- Target function must be picklable (defined at module level, not inside another function)
- Avoid passing unpicklable objects (e.g., file handles, network connections)
- Consider using ThreadTask for unpicklable objects

### GIL-related Performance Issues

**Problem**: ThreadTask doesn't provide speedup for CPU-bound work

**Solution**:
- Switch to ProcessTask for CPU-bound operations
- ThreadTask is only effective for I/O-bound work

## Integration with ModularNet

The Task system integrates with ModularNet's plugin architecture, which uses multiprocessing for plugin isolation. While Task provides low-level parallel execution, the plugin system builds on it for higher-level concurrency.

**See Also**:
- [Plugin System Documentation](../../plugin/docs/PLUGIN_SYSTEM_COMPREHENSIVE.md)
- [Trial Management](../../trial/docs/)

## API Reference

### Imports

```python
from modularnet.task import BaseTask, ThreadTask, ProcessTask, Task
```

### Quick Reference

```python
# Create tasks
thread_task = ThreadTask(fn, args=(arg1,), kwargs={'key': 'val'})
process_task = ProcessTask(fn, args=(arg1,), kwargs={'key': 'val'})
task = Task.create(fn, mode="thread", args=(arg1,), kwargs={'key': 'val'})

# Control tasks
task.start()
task.join()
is_running = task.is_alive()
result = task.run()  # Execute synchronously
```

## Version History

- **1.0.0**: Initial implementation with ThreadTask, ProcessTask, and Task factory
