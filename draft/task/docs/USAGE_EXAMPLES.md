# Task System Usage Guide

Practical examples and recipes for using the ModularNet Task system.

## Quick Start

### Example 1: Simple Thread Task

```python
from modularnet.task import ThreadTask
import time

def greet(name, delay=0):
    """Simulate a greeting function that takes some time."""
    if delay:
        time.sleep(delay)
    return f"Hello, {name}!"

# Create and execute a task
task = ThreadTask(greet, args=("Alice",), kwargs={'delay': 1})
task.start()
print("Task started...")
task.join()
print("Task completed!")
```

### Example 2: Simple Process Task

```python
from modularnet.task import ProcessTask

def compute_sum(numbers):
    """Sum a list of numbers."""
    return sum(numbers)

# Create and execute a process task
task = ProcessTask(compute_sum, args=(list(range(1000000)),))
task.start()
task.join()
print("Computation completed!")
```

### Example 3: Using the Factory

```python
from modularnet.task import Task

def process_data(data):
    """Process data."""
    return len(data)

# Choose mode dynamically
mode = "process" if len(data) > 100000 else "thread"
task = Task.create(process_data, mode=mode, args=(data,))
task.start()
task.join()
```

## Common Patterns

### Pattern 1: Monitor Task Progress

```python
from modularnet.task import ThreadTask
import time

def long_running_task():
    """Simulate a long-running task."""
    for i in range(10):
        print(f"Step {i}")
        time.sleep(1)

task = ThreadTask(long_running_task)
task.start()

# Monitor the task
while task.is_alive():
    print("Task is still running...")
    time.sleep(2)

print("Task finished!")
task.join()
```

### Pattern 2: Parallel Data Processing

```python
from modularnet.task import ProcessTask
import time

def process_batch(batch_id, batch_size):
    """Process a batch of data."""
    print(f"Processing batch {batch_id} with {batch_size} items")
    time.sleep(1)  # Simulate processing
    return batch_id * batch_size

# Create tasks for multiple batches
tasks = [
    ProcessTask(process_batch, args=(i, 1000))
    for i in range(4)
]

# Start all tasks
start_time = time.time()
for task in tasks:
    task.start()

# Wait for completion
for task in tasks:
    task.join()

elapsed = time.time() - start_time
print(f"Processed 4 batches in {elapsed:.1f} seconds")
```

### Pattern 3: Fetch Multiple URLs (I/O-Bound)

```python
from modularnet.task import ThreadTask
import requests
import time

def fetch_url(url, timeout=5):
    """Fetch content from URL."""
    try:
        response = requests.get(url, timeout=timeout)
        return f"{url}: {response.status_code}"
    except Exception as e:
        return f"{url}: ERROR - {e}"

urls = [
    "https://example.com",
    "https://github.com",
    "https://python.org",
    "https://wikipedia.org"
]

# Create thread tasks (great for I/O operations)
tasks = [ThreadTask(fetch_url, args=(url,)) for url in urls]

# Start all tasks
start_time = time.time()
for task in tasks:
    task.start()

# Wait for all to complete
for task in tasks:
    task.join()

elapsed = time.time() - start_time
print(f"Fetched {len(urls)} URLs in {elapsed:.1f} seconds")
```

### Pattern 4: Shared Memory with Threads

```python
from modularnet.task import ThreadTask
import time

class DataProcessor:
    """Process data with shared state."""
    
    def __init__(self):
        self.results = []
    
    def process_item(self, item_id, delay=0.1):
        """Process a single item."""
        if delay:
            time.sleep(delay)
        self.results.append(f"Processed {item_id}")

# Create processor
processor = DataProcessor()

# Create tasks that share the processor's memory
tasks = [
    ThreadTask(processor.process_item, args=(i, 0.1))
    for i in range(5)
]

# Start all tasks
for task in tasks:
    task.start()

# Wait for completion
for task in tasks:
    task.join()

# Access shared results
print(f"Results: {processor.results}")
```

### Pattern 5: Error Handling

```python
from modularnet.task import ThreadTask

def risky_operation(value):
    """Perform an operation that might fail."""
    if value < 0:
        raise ValueError("Value must be non-negative")
    return value ** 2

# Synchronous execution to catch exceptions
task = ThreadTask(risky_operation, args=(-5,))
try:
    result = task.run()
except ValueError as e:
    print(f"Error: {e}")

# Asynchronous execution
task2 = ThreadTask(risky_operation, args=(5,))
task2.start()
try:
    task2.join()
except Exception as e:
    print(f"Task error: {e}")
```

### Pattern 6: Worker Pool for ML Training

```python
from modularnet.task import ProcessTask
import time

def train_model(model_id, epochs=10):
    """Simulate model training."""
    print(f"Model {model_id}: Training for {epochs} epochs...")
    time.sleep(2)  # Simulate training
    accuracy = 0.9 + (model_id * 0.01)
    return f"Model {model_id}: {accuracy:.2%} accuracy"

# Train multiple models in parallel
num_models = 4
tasks = [
    ProcessTask(train_model, args=(i, 10))
    for i in range(num_models)
]

print(f"Starting training of {num_models} models...")
start_time = time.time()

for task in tasks:
    task.start()

for task in tasks:
    task.join()

elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.1f} seconds")
```

### Pattern 7: Sequential Task Chain

```python
from modularnet.task import ThreadTask
import time

def fetch_data():
    """Fetch data from source."""
    time.sleep(1)
    return [1, 2, 3, 4, 5]

def transform_data(data):
    """Transform the data."""
    time.sleep(1)
    return [x * 2 for x in data]

def save_data(data):
    """Save the transformed data."""
    time.sleep(1)
    print(f"Saved: {data}")
    return True

# Execute tasks sequentially
def pipeline():
    # Step 1: Fetch
    task1 = ThreadTask(fetch_data)
    task1.start()
    task1.join()
    data = task1.run()
    
    # Step 2: Transform
    task2 = ThreadTask(transform_data, args=(data,))
    task2.start()
    task2.join()
    transformed = task2.run()
    
    # Step 3: Save
    task3 = ThreadTask(save_data, args=(transformed,))
    task3.start()
    task3.join()

pipeline()
```

## When to Use Each Type

### Use ThreadTask When:

- ✅ Performing I/O operations (network requests, file operations)
- ✅ Database queries or API calls
- ✅ The operations need to share memory with parent
- ✅ You need quick concurrent execution
- ✅ Working with JSON, API responses, etc.

**Example**:
```python
from modularnet.task import ThreadTask
import requests

task = ThreadTask(requests.get, args=("https://api.github.com",))
task.start()
task.join()
```

### Use ProcessTask When:

- ✅ Performing CPU-intensive calculations
- ✅ Machine learning model training or inference
- ✅ Data processing and transformation
- ✅ Scientific computing
- ✅ You need true parallelism (no GIL limitations)

**Example**:
```python
from modularnet.task import ProcessTask
import numpy as np

def matrix_multiply(size):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    return np.dot(A, B)

task = ProcessTask(matrix_multiply, args=(1000,))
task.start()
task.join()
```

## Performance Considerations

### Thread Task Performance

```python
# I/O-bound operation benefits greatly from threading
import time
from modularnet.task import ThreadTask

def slow_api_call():
    time.sleep(2)  # Simulate API delay
    return "data"

# Sequential: ~8 seconds
tasks = [ThreadTask(slow_api_call) for _ in range(4)]
for task in tasks:
    task.run()

# Parallel: ~2 seconds
tasks = [ThreadTask(slow_api_call) for _ in range(4)]
for task in tasks:
    task.start()
for task in tasks:
    task.join()
```

### Process Task Performance

```python
# CPU-bound operation benefits greatly from processes
import time
from modularnet.task import ProcessTask

def cpu_work(n):
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# Sequential: ~4 seconds (2 tasks × 2 seconds each)
tasks = [ProcessTask(cpu_work, args=(10**8,)) for _ in range(2)]
for task in tasks:
    task.run()

# Parallel: ~2 seconds (true parallelism)
tasks = [ProcessTask(cpu_work, args=(10**8,)) for _ in range(2)]
for task in tasks:
    task.start()
for task in tasks:
    task.join()
```

## Troubleshooting

### Issue: "RuntimeError: cannot join thread before it is started"

**Problem**: Calling `join()` before `start()`

**Solution**: Always call `start()` before `join()`
```python
task = ThreadTask(my_func)
task.start()  # Must start first
task.join()   # Then join
```

### Issue: ProcessTask seems slow

**Problem**: Process creation overhead

**Solution**: Use threads for I/O, processes only for CPU-bound work
```python
# Wrong: process overhead not worth it
task = ProcessTask(requests.get, args=(url,))

# Correct: thread is better for I/O
task = ThreadTask(requests.get, args=(url,))
```

### Issue: Task appears to hang

**Problem**: Target function has infinite loop or blocks indefinitely

**Solution**: Debug the target function
```python
# Test the function directly
result = my_function()  # Check if it returns

# Add timeout monitoring
task.start()
start = time.time()
while task.is_alive():
    if time.time() - start > 30:
        print("Task seems stuck")
        break
```

### Issue: PicklingError with ProcessTask

**Problem**: Target function or arguments can't be pickled

**Solution**: Ensure target is defined at module level (not nested)
```python
# Bad: nested function can't be pickled
def outer():
    def inner():  # This can't be pickled!
        pass
    return ProcessTask(inner)

# Good: module-level function is picklable
def my_function():  # Defined at module level
    pass

task = ProcessTask(my_function)
```

## Best Practices Checklist

- ✅ Always call `join()` before program exits to wait for task completion
- ✅ Use `is_alive()` to monitor task status
- ✅ Choose ThreadTask for I/O-bound work, ProcessTask for CPU-bound
- ✅ Test with both modes during development
- ✅ Handle exceptions in task targets
- ✅ For large data transfers, consider using multiprocessing.Queue
- ✅ Use thread-safe data structures with ThreadTask (Queue, Lock, etc.)
- ✅ Profile your code to ensure chosen mode provides speedup

## Integration Examples

### With PyTorch

```python
from modularnet.task import ProcessTask
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

def train_model(model_path, epochs):
    model = SimpleModel()
    # ... training code ...
    return accuracy

task = ProcessTask(train_model, args=("model.pth", 100))
task.start()
task.join()
```

### With pandas

```python
from modularnet.task import ThreadTask
import pandas as pd

def load_and_process_csv(filepath):
    df = pd.read_csv(filepath)
    df['value'] = df['value'] * 2
    return df.describe()

task = ThreadTask(load_and_process_csv, args=("data.csv",))
task.start()
task.join()
```

## See Also

- [Task System Documentation](TASK_SYSTEM.md)
- [BaseTask API Reference](TASK_SYSTEM.md#basetask)
- [ThreadTask Guide](TASK_SYSTEM.md#threadtask)
- [ProcessTask Guide](TASK_SYSTEM.md#processtask)
