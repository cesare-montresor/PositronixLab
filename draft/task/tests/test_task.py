"""Comprehensive tests for the ModularNet Task system.

Tests cover:
- ThreadTask: Threading-based task execution
- ProcessTask: Multiprocessing-based task execution
- Task factory: Correct instantiation based on mode
- Lifecycle: start(), join(), is_alive()
- Edge cases: None kwargs, empty args, complex functions
- Concurrent execution: Multiple tasks running in parallel
- Error handling: Invalid modes, invalid targets
"""

import unittest
import time
import threading
import multiprocessing
from draft.task import BaseTask, ThreadTask, ProcessTask, Task


# Test helper functions
def simple_function(value=10):
    """Simple function that returns the value."""
    return value


def function_with_args(a, b):
    """Function that takes two arguments."""
    return a + b


def function_with_kwargs(x, y=5, z=10):
    """Function that takes positional and keyword arguments."""
    return x + y + z


def function_with_side_effects(results_list, index, value):
    """Function that modifies an external list (for thread testing)."""
    results_list[index] = value


def slow_function(duration=1):
    """Function that takes time to execute."""
    time.sleep(duration)
    return f"Completed after {duration} seconds"


def cpu_intensive_function(n=100):
    """CPU-intensive function for process testing."""
    total = 0
    for i in range(n):
        total += i ** 2
    return total


class TestThreadTask(unittest.TestCase):
    """Test cases for ThreadTask class."""

    def test_thread_task_creation(self):
        """Test ThreadTask can be created with target function."""
        task = ThreadTask(simple_function)
        self.assertIsNotNone(task)
        self.assertIsInstance(task, BaseTask)
        self.assertIsNotNone(task.thread)

    def test_thread_task_with_args(self):
        """Test ThreadTask with positional arguments."""
        task = ThreadTask(function_with_args, args=(5, 3))
        self.assertIsNotNone(task)
        self.assertEqual(task.args, (5, 3))

    def test_thread_task_with_kwargs(self):
        """Test ThreadTask with keyword arguments."""
        kwargs = {'y': 20, 'z': 30}
        task = ThreadTask(function_with_kwargs, args=(10,), kwargs=kwargs)
        self.assertIsNotNone(task)
        self.assertEqual(task.kwargs, kwargs)

    def test_thread_task_with_none_kwargs(self):
        """Test ThreadTask handles None kwargs correctly."""
        task = ThreadTask(simple_function, kwargs=None)
        self.assertIsNotNone(task)
        self.assertEqual(task.kwargs, {})

    def test_thread_task_start_and_join(self):
        """Test ThreadTask start() and join() lifecycle."""
        task = ThreadTask(simple_function)
        self.assertFalse(task.is_alive())
        
        task.start()
        # Thread might be very quick, so we check the general flow
        
        task.join()
        self.assertFalse(task.is_alive())

    def test_thread_task_is_alive(self):
        """Test ThreadTask is_alive() status tracking."""
        task = ThreadTask(slow_function, args=(0.2,))
        self.assertFalse(task.is_alive())
        
        task.start()
        self.assertTrue(task.is_alive())
        
        task.join()
        self.assertFalse(task.is_alive())

    def test_thread_task_run(self):
        """Test ThreadTask run() executes the target function."""
        task = ThreadTask(simple_function, args=(42,))
        result = task.run()
        self.assertEqual(result, 42)

    def test_thread_task_with_multiple_args(self):
        """Test ThreadTask with multiple positional arguments."""
        task = ThreadTask(function_with_args, args=(10, 20))
        result = task.run()
        self.assertEqual(result, 30)

    def test_thread_task_with_mixed_args_kwargs(self):
        """Test ThreadTask with both positional and keyword arguments."""
        task = ThreadTask(
            function_with_kwargs,
            args=(5,),
            kwargs={'y': 15, 'z': 25}
        )
        result = task.run()
        self.assertEqual(result, 5 + 15 + 25)

    def test_thread_task_side_effects(self):
        """Test ThreadTask can modify shared memory (key thread feature)."""
        results = [None, None, None]
        
        task = ThreadTask(
            function_with_side_effects,
            args=(results, 0, "thread_works")
        )
        task.start()
        task.join()
        
        self.assertEqual(results[0], "thread_works")

    def test_thread_task_concurrent_execution(self):
        """Test multiple ThreadTasks execute concurrently."""
        start_time = time.time()
        
        tasks = [
            ThreadTask(slow_function, args=(0.1,)) for _ in range(3)
        ]
        
        # Start all tasks
        for task in tasks:
            task.start()
        
        # Wait for all
        for task in tasks:
            task.join()
        
        elapsed = time.time() - start_time
        
        # Should take ~0.1s for concurrent execution, not 0.3s sequential
        self.assertLess(elapsed, 0.3)


class TestProcessTask(unittest.TestCase):
    """Test cases for ProcessTask class."""

    def test_process_task_creation(self):
        """Test ProcessTask can be created with target function."""
        task = ProcessTask(simple_function)
        self.assertIsNotNone(task)
        self.assertIsInstance(task, BaseTask)
        self.assertIsNotNone(task.process)

    def test_process_task_with_args(self):
        """Test ProcessTask with positional arguments."""
        task = ProcessTask(function_with_args, args=(5, 3))
        self.assertIsNotNone(task)
        self.assertEqual(task.args, (5, 3))

    def test_process_task_with_kwargs(self):
        """Test ProcessTask with keyword arguments."""
        kwargs = {'y': 20, 'z': 30}
        task = ProcessTask(function_with_kwargs, args=(10,), kwargs=kwargs)
        self.assertIsNotNone(task)
        self.assertEqual(task.kwargs, kwargs)

    def test_process_task_with_none_kwargs(self):
        """Test ProcessTask handles None kwargs correctly."""
        task = ProcessTask(simple_function, kwargs=None)
        self.assertIsNotNone(task)
        self.assertEqual(task.kwargs, {})

    def test_process_task_start_and_join(self):
        """Test ProcessTask start() and join() lifecycle."""
        task = ProcessTask(simple_function)
        self.assertFalse(task.is_alive())
        
        task.start()
        # Process might complete quickly
        
        task.join()
        self.assertFalse(task.is_alive())

    def test_process_task_is_alive(self):
        """Test ProcessTask is_alive() status tracking."""
        task = ProcessTask(slow_function, args=(0.3,))
        self.assertFalse(task.is_alive())
        
        task.start()
        self.assertTrue(task.is_alive())
        
        task.join()
        self.assertFalse(task.is_alive())

    def test_process_task_run(self):
        """Test ProcessTask run() executes the target function."""
        task = ProcessTask(simple_function, args=(42,))
        result = task.run()
        self.assertEqual(result, 42)

    def test_process_task_with_multiple_args(self):
        """Test ProcessTask with multiple positional arguments."""
        task = ProcessTask(function_with_args, args=(10, 20))
        result = task.run()
        self.assertEqual(result, 30)

    def test_process_task_with_mixed_args_kwargs(self):
        """Test ProcessTask with both positional and keyword arguments."""
        task = ProcessTask(
            function_with_kwargs,
            args=(5,),
            kwargs={'y': 15, 'z': 25}
        )
        result = task.run()
        self.assertEqual(result, 5 + 15 + 25)

    def test_process_task_cpu_intensive(self):
        """Test ProcessTask for CPU-intensive operations."""
        task = ProcessTask(cpu_intensive_function, args=(1000,))
        task.start()
        task.join()
        
        # Should complete without hanging
        self.assertFalse(task.is_alive())

    def test_process_task_process_isolation(self):
        """Test ProcessTask provides process isolation."""
        # The process should be separate
        task = ProcessTask(simple_function, args=(99,))
        self.assertIsNotNone(task.process)
        # PID is None until process is started
        task.start()
        self.assertIsNotNone(task.process.pid)
        task.join()


class TestTaskFactory(unittest.TestCase):
    """Test cases for Task factory class."""

    def test_factory_creates_thread_task(self):
        """Test Task.create() creates ThreadTask with mode='thread'."""
        task = Task.create(simple_function, mode="thread")
        self.assertIsInstance(task, ThreadTask)

    def test_factory_creates_process_task(self):
        """Test Task.create() creates ProcessTask with mode='process'."""
        task = Task.create(simple_function, mode="process")
        self.assertIsInstance(task, ProcessTask)

    def test_factory_default_mode_is_thread(self):
        """Test Task.create() defaults to thread mode."""
        task = Task.create(simple_function)
        self.assertIsInstance(task, ThreadTask)

    def test_factory_with_args(self):
        """Test Task.create() passes args correctly."""
        task = Task.create(
            function_with_args,
            mode="thread",
            args=(5, 10)
        )
        self.assertIsInstance(task, ThreadTask)
        self.assertEqual(task.args, (5, 10))

    def test_factory_with_kwargs(self):
        """Test Task.create() passes kwargs correctly."""
        kwargs = {'y': 20, 'z': 30}
        task = Task.create(
            function_with_kwargs,
            mode="thread",
            args=(10,),
            kwargs=kwargs
        )
        self.assertIsInstance(task, ThreadTask)
        self.assertEqual(task.kwargs, kwargs)

    def test_factory_with_invalid_mode(self):
        """Test Task.create() raises ValueError for invalid mode."""
        with self.assertRaises(ValueError) as context:
            Task.create(simple_function, mode="invalid")
        
        self.assertIn("thread", str(context.exception))
        self.assertIn("process", str(context.exception))

    def test_factory_with_empty_string_mode(self):
        """Test Task.create() raises ValueError for empty mode."""
        with self.assertRaises(ValueError):
            Task.create(simple_function, mode="")

    def test_factory_mode_case_sensitive(self):
        """Test Task.create() mode parameter is case-sensitive."""
        with self.assertRaises(ValueError):
            Task.create(simple_function, mode="Thread")
        
        with self.assertRaises(ValueError):
            Task.create(simple_function, mode="PROCESS")

    def test_factory_creates_independent_instances(self):
        """Test Task.create() creates independent task instances."""
        task1 = Task.create(simple_function)
        task2 = Task.create(simple_function)
        
        self.assertIsNot(task1, task2)


class TestTaskLifecycle(unittest.TestCase):
    """Test cases for task lifecycle management."""

    def test_thread_task_lifecycle(self):
        """Test complete lifecycle of ThreadTask."""
        task = ThreadTask(slow_function, args=(0.1,))
        
        # Initial state
        self.assertFalse(task.is_alive())
        
        # After start
        task.start()
        self.assertTrue(task.is_alive())
        
        # After join
        task.join()
        self.assertFalse(task.is_alive())

    def test_process_task_lifecycle(self):
        """Test complete lifecycle of ProcessTask."""
        task = ProcessTask(slow_function, args=(0.1,))
        
        # Initial state
        self.assertFalse(task.is_alive())
        
        # After start
        task.start()
        self.assertTrue(task.is_alive())
        
        # After join
        task.join()
        self.assertFalse(task.is_alive())

    def test_multiple_start_calls_thread(self):
        """Test calling start() multiple times on ThreadTask."""
        task = ThreadTask(simple_function)
        task.start()
        
        # Calling start again should raise RuntimeError
        with self.assertRaises(RuntimeError):
            task.start()
        
        task.join()

    def test_multiple_start_calls_process(self):
        """Test calling start() multiple times on ProcessTask."""
        task = ProcessTask(simple_function)
        task.start()
        
        # Calling start again should raise ValueError or AssertionError
        with self.assertRaises((ValueError, AssertionError, RuntimeError)):
            task.start()
        
        task.join()

    def test_join_without_start_thread(self):
        """Test calling join() without start() on ThreadTask."""
        task = ThreadTask(simple_function)
        
        # Calling join() on a non-started thread raises RuntimeError
        with self.assertRaises(RuntimeError):
            task.join()


class TestConcurrentExecution(unittest.TestCase):
    """Test cases for concurrent execution of multiple tasks."""

    def test_concurrent_thread_tasks(self):
        """Test multiple ThreadTasks execute concurrently."""
        start_time = time.time()
        
        tasks = [
            ThreadTask(slow_function, args=(0.1,)) for _ in range(5)
        ]
        
        for task in tasks:
            task.start()
        
        for task in tasks:
            task.join()
        
        elapsed = time.time() - start_time
        
        # Should take ~0.1s, not 0.5s
        self.assertLess(elapsed, 0.4)

    def test_concurrent_process_tasks(self):
        """Test multiple ProcessTasks execute in parallel."""
        start_time = time.time()
        
        tasks = [
            ProcessTask(slow_function, args=(0.15,)) for _ in range(3)
        ]
        
        for task in tasks:
            task.start()
        
        for task in tasks:
            task.join()
        
        elapsed = time.time() - start_time
        
        # Should take ~0.15s minimum, could be more depending on system
        # but not 0.45s (sequential)
        self.assertLess(elapsed, 0.5)

    def test_mixed_concurrent_tasks(self):
        """Test ThreadTasks and ProcessTasks can run together."""
        thread_task = ThreadTask(slow_function, args=(0.1,))
        process_task = ProcessTask(slow_function, args=(0.1,))
        
        start_time = time.time()
        
        thread_task.start()
        process_task.start()
        
        thread_task.join()
        process_task.join()
        
        elapsed = time.time() - start_time
        
        # Should be concurrent, not sequential
        self.assertLess(elapsed, 0.3)

    def test_task_status_while_running(self):
        """Test is_alive() status while tasks are running."""
        task = ThreadTask(slow_function, args=(0.2,))
        task.start()
        
        alive_count = 0
        for _ in range(10):
            if task.is_alive():
                alive_count += 1
            time.sleep(0.05)
        
        task.join()
        
        # Should have been alive at least once
        self.assertGreater(alive_count, 0)
        self.assertFalse(task.is_alive())


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and special scenarios."""

    def test_empty_args(self):
        """Test task with no arguments."""
        task = ThreadTask(simple_function)
        result = task.run()
        self.assertEqual(result, 10)  # Default value

    def test_empty_kwargs(self):
        """Test task with empty kwargs dict."""
        task = ThreadTask(function_with_kwargs, args=(5,), kwargs={})
        result = task.run()
        self.assertEqual(result, 5 + 5 + 10)  # Using defaults

    def test_target_with_no_return_value(self):
        """Test task target that doesn't return anything."""
        def no_return_function():
            x = 1 + 1
        
        task = ThreadTask(no_return_function)
        result = task.run()
        self.assertIsNone(result)

    def test_target_with_exception(self):
        """Test task target that raises exception."""
        def exception_function():
            raise ValueError("Test exception")
        
        task = ThreadTask(exception_function)
        
        with self.assertRaises(ValueError):
            task.run()

    def test_large_args_process_task(self):
        """Test ProcessTask with large arguments."""
        large_list = list(range(10000))
        
        def sum_list(lst):
            return sum(lst)
        
        task = ProcessTask(sum_list, args=(large_list,))
        result = task.run()
        self.assertEqual(result, sum(large_list))

    def test_kwargs_override_defaults(self):
        """Test that kwargs properly override defaults."""
        task = ThreadTask(
            function_with_kwargs,
            args=(1,),
            kwargs={'y': 100, 'z': 1000}
        )
        result = task.run()
        self.assertEqual(result, 1 + 100 + 1000)

    def test_special_characters_in_kwargs(self):
        """Test task with special characters in string kwargs."""
        def string_function(text):
            return text
        
        special_string = "Test with special chars: @#$%^&*()"
        task = ThreadTask(string_function, args=(special_string,))
        result = task.run()
        self.assertEqual(result, special_string)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests simulating real-world scenarios."""

    def test_producer_consumer_with_threads(self):
        """Test producer-consumer pattern with threads."""
        data_container = {'data': None}
        
        def producer():
            time.sleep(0.05)
            data_container['data'] = 42
        
        def consumer():
            # Wait for data to be available
            timeout = 1
            start = time.time()
            while data_container['data'] is None and time.time() - start < timeout:
                time.sleep(0.01)
            return data_container['data']
        
        producer_task = ThreadTask(producer)
        consumer_task = ThreadTask(consumer)
        
        producer_task.start()
        # Give producer time to set data
        producer_task.join()
        
        consumer_task.start()
        consumer_task.join()
        
        self.assertEqual(data_container['data'], 42)

    def test_batch_processing(self):
        """Test batch processing with multiple tasks."""
        def process_batch(batch_num, batch_size):
            total = 0
            for i in range(batch_size):
                total += batch_num * batch_size + i
            return total
        
        tasks = [
            ProcessTask(process_batch, args=(i, 100))
            for i in range(5)
        ]
        
        for task in tasks:
            task.start()
        
        for task in tasks:
            task.join()
        
        # All tasks should complete successfully
        for task in tasks:
            self.assertFalse(task.is_alive())

    def test_worker_pool_simulation(self):
        """Simulate a worker pool pattern."""
        def worker(worker_id, task_list):
            total = 0
            for task in task_list:
                total += task
            return f"Worker {worker_id}: {total}"
        
        tasks_per_worker = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        workers = [
            ThreadTask(worker, args=(i, tasks))
            for i, tasks in enumerate(tasks_per_worker)
        ]
        
        for worker in workers:
            worker.start()
        
        for worker in workers:
            worker.join()
        
        # All workers should complete
        for worker in workers:
            self.assertFalse(worker.is_alive())


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and validation."""

    def test_invalid_mode_error_message(self):
        """Test that invalid mode provides helpful error message."""
        with self.assertRaises(ValueError) as context:
            Task.create(simple_function, mode="gpu")
        
        error_msg = str(context.exception)
        self.assertIn("Mode", error_msg)
        self.assertIn("thread", error_msg)
        self.assertIn("process", error_msg)

    def test_none_target_function(self):
        """Test that None target is handled."""
        with self.assertRaises(TypeError):
            task = ThreadTask(None)
            task.run()

    def test_non_callable_target(self):
        """Test that non-callable target raises error."""
        with self.assertRaises(TypeError):
            task = ThreadTask("not a function")
            task.run()


if __name__ == '__main__':
    unittest.main()
