# Logger Usage Guide

The rewritten `Logger` class provides robust multithreading and multiprocessing support with automatic stdout/stderr capturing.

## Quick Start (One-liner)

```python
from positronixlab.common.logger import Logger.capture

# Activate logger for current thread/subprocess
logger = Logger.capture(name='my_task')
logger.log('Hello world')
print("This is captured automatically")
```

## Basic Logging Methods

```python
from positronixlab.common.logger import Logger, Logger.capture

logger = Logger.capture()

# Different log levels
logger.debug('Debug message')      # [DEBUG]
logger.log('Info message')         # [LOG]
logger.info('Info message')        # [LOG] (alias)
logger.warning('Warning message')  # [WARNING]
logger.error('Error message')      # [ERROR]
```

## Example: Multithreading

```python
from positronixlab.common.logger import Logger.capture
import threading

def worker_task(worker_id):
    # Each thread gets its own logger context
    logger = Logger.capture(name=f'worker_{worker_id}')
    logger.log(f'Worker {worker_id} started')
    print(f"Worker {worker_id} doing work...")  # Captured automatically
    logger.log(f'Worker {worker_id} done')

# Start threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker_task, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

**Log files created:**
- `logs/positronixlab_pid<PID>_threadworker_0_20260417.log`
- `logs/positronixlab_pid<PID>_threadworker_1_20260417.log`
- `logs/positronixlab_pid<PID>_threadworker_2_20260417.log`

## Example: Multiprocessing

```python
from positronixlab.common.logger import Logger.capture
import multiprocessing

def process_task(task_id):
    # Each process gets its own logger context
    logger = Logger.capture(name=f'process_{task_id}')
    logger.log(f'Process {task_id} started')
    print(f"Processing task {task_id}...")  # Captured automatically
    logger.log(f'Process {task_id} complete')

if __name__ == '__main__':
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=process_task, args=(i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
```

**Log files created (per process with unique PID):**
- `logs/positronixlab_pid<PID1>_threadprocess_0_20260417.log`
- `logs/positronixlab_pid<PID2>_threadprocess_1_20260417.log`
- `logs/positronixlab_pid<PID3>_threadprocess_2_20260417.log`

## Advanced Usage

### Selective Capture

```python
logger = Logger(
    base_path='./logs',
    name='myapp',
    capture_stdout=True,   # Capture print() statements
    capture_stderr=False   # Don't capture stderr
)
```

### Context Manager for Temporary Capture

```python
logger = Logger.capture()

logger.log('Starting')
print("Not captured initially")

# Capture only within this block
with logger.capture_context(stdout=True, stderr=True):
    print("This is captured")
    logger.log("Inside context")

print("Back to not captured")
```

### Manual Capture Control

```python
logger = Logger.capture(capture_stdout=False)

# Manually enable later
logger.enable_capture(stdout=True, stderr=True)
print("Now captured")

# Disable capture
logger.disable_capture()
print("No longer captured")
```

### Cleanup

```python
logger = Logger.capture()
# ... do work ...

# Flush all log files
logger.flush()

# Close all log files
logger.close()
```

## Log File Structure

Each log file is named with the following pattern:
```
{name}_pid{process_id}_thread{thread_name}_{YYYYMMDD}.log
```

Log messages are formatted as:
```
[YYYY-MM-DD HH:MM:SS] [LEVEL    ] [context_id] message
```

Example:
```
[2026-04-17 18:57:12] [LOG     ] [myapp_pid363747_threadMainThread] My message
[2026-04-17 18:57:12] [WARNING ] [myapp_pid363747_threadMainThread] Warning!
[2026-04-17 18:57:12] [STDOUT  ] [myapp_pid363747_threadMainThread] print() output
```

## Key Features

✅ **Thread-safe**: Each thread/process gets isolated log files  
✅ **Process-safe**: Different processes (different PIDs) logged separately  
✅ **Automatic stdout/stderr capture**: Optional capturing of print() statements  
✅ **One-liner activation**: Simple `Logger.capture()` call  
✅ **Multiple log levels**: DEBUG, LOG, WARNING, ERROR  
✅ **Meaningful context names**: Process ID, thread name in all logs  
✅ **Context managers**: Temporary capture control  
✅ **No external dependencies**: Only uses stdlib (no extra requirements)

## Ignored Patterns

The following messages are automatically filtered when captured:
- AxParameterWarning
- repr attribute warnings
- Full backward hook warnings

Add more patterns by modifying `logger.ignore_patterns` if needed.

## Troubleshooting

**No logs being created?**
- Ensure `base_path` directory is writable
- Check `show_console=True` to see errors

**stdout/stderr not captured?**
- Enable with `capture_stdout=True` and/or `capture_stderr=True`
- Or use `logger.enable_capture()` method

**Lost logs when subprocess crashes?**
- Call `logger.flush()` before critical operations
- Or use context manager for automatic cleanup
