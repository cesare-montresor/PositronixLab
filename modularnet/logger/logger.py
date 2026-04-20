
from __future__ import annotations

import os
import sys
import threading
import multiprocessing
import queue
import time
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, TextIO
import warnings

warnings.filterwarnings('ignore', message='.*Full backward hook is firing.*')
warnings.filterwarnings('ignore', category=UserWarning)


class StreamCapture:
    """Wraps a file object to capture writes while allowing passthrough"""
    def __init__(self, original_stream, logger_instance):
        self.original = original_stream
        self.logger = logger_instance
        self.isatty = original_stream.isatty if hasattr(original_stream, 'isatty') else lambda: False
        
    def write(self, text):
        if text and text.strip():
            self.logger._queue_log(text.rstrip(), 'STDOUT')
        return self.original.write(text)
    
    def flush(self):
        return self.original.flush()
    
    def __getattr__(self, name):
        return getattr(self.original, name)


class Logger:
    """Thread-safe and process-safe logger with stdout/stderr capturing"""
    
    LEVELS = ['DEBUG', 'LOG', 'WARNING', 'ERROR']
    _instance = None
    _lock = threading.Lock()


    @staticmethod
    def capture(base_path: str = './logs',
                 name: str = 'modularnet',
                 capture_stdout: bool = True,
                 capture_stderr: bool = True) -> Logger:
        """
        One-liner to activate logger for current thread/subprocess with stdout/stderr capture.
        
        Usage:
            # In main thread
            logger = capture()
            
            # In a subprocess target function
            def worker():
                logger = Logger.capture(name='worker_task')
                print("This is captured")
            
            # In a thread target
            def thread_worker():
                logger = Logger.capture(name='background_task')
        
        Args:
            base_path: Directory for logs
            name: Name for this logger context
            capture_stdout: Capture stdout
            capture_stderr: Capture stderr
        
        Returns:
            Logger instance configured and ready to use
        """
        logger = Logger(
            base_path=base_path,
            name=name,
            show_console=True,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr
        )
        return logger



    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, 
                 base_path: str = './logs',
                 name: str = 'modularnet',
                 backend: str = 'file',
                 show_console: bool = True,
                 show_level: str = 'LOG',
                 capture_stdout: bool = False,
                 capture_stderr: bool = False):
        """
        Initialize Logger with multithread/multiprocess support.
        
        Args:
            base_path: Directory to store log files
            name: Base name for logs
            backend: 'file' or 'wandb'
            show_console: Print to console
            show_level: Minimum level to display (DEBUG, LOG, WARNING, ERROR)
            capture_stdout: Capture stdout
            capture_stderr: Capture stderr
        """
        if self._initialized:
            return
        
        self._initialized = True
        self.base_path = base_path
        self.name = name
        self.backend = backend
        self.show_console = show_console
        self.show_level = show_level
        self.capture_stdout_flag = capture_stdout
        self.capture_stderr_flag = capture_stderr
        
        # Thread/process tracking
        self.files: Dict[str, TextIO] = {}  # context_id -> file handle
        self.lock = threading.RLock()
        self.main_pid = os.getpid()
        self.main_thread = threading.current_thread()
        
        # Store original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Ignore patterns for captured output
        self.ignore_patterns = [
            re.compile(r'.*AxParameterWarning.*'),
            re.compile(r'.*repr.*attribute.*'),
            re.compile(r'.*Full backward hook.*'),
        ]
        
        self._init_backend()
        
        if capture_stdout or capture_stderr:
            self.enable_capture(stdout=capture_stdout, stderr=capture_stderr)

    def _init_backend(self):
        """Initialize the logging backend"""
        os.makedirs(self.base_path, exist_ok=True)
        
        if self.backend == 'file':
            pass  # Already created directory
        elif self.backend == 'wandb':
            try:
                import wandb
                wandb.init(project=self.name, mode='online')
            except Exception as e:
                self.original_stdout.write(f"Warning: Could not initialize wandb: {e}\n")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _get_context_id(self) -> str:
        """Get unique identifier for current thread/process context"""
        pid = os.getpid()
        thread_name = threading.current_thread().name
        return f"{self.name}_pid{pid}_thread{thread_name}"

    def _get_log_filepath(self, context_id: str) -> str:
        """Get the log file path for a context"""
        safe_id = context_id.replace('/', '_').replace('\\', '_')
        timestamp = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.base_path, f"{safe_id}_{timestamp}.log")

    def _should_ignore(self, text: str) -> bool:
        """Check if text matches ignore patterns"""
        return any(pat.search(text) for pat in self.ignore_patterns)

    def _get_or_create_file(self, context_id: str) -> TextIO:
        """Get or create file handle for context"""
        if context_id not in self.files:
            with self.lock:
                if context_id not in self.files:
                    filepath = self._get_log_filepath(context_id)
                    self.files[context_id] = open(filepath, 'a', buffering=1)
                    header = f"\n{'='*80}\n"
                    header += f"Log session started: {datetime.now().isoformat()}\n"
                    header += f"Process ID: {os.getpid()}, Thread: {threading.current_thread().name}\n"
                    header += f"{'='*80}\n"
                    self.files[context_id].write(header)
        return self.files[context_id]

    def _queue_log(self, message: str, level: str = 'LOG'):
        """Internal method to log a message (used by captured output)"""
        if self._should_ignore(message):
            return
        
        if level == 'STDOUT':
            if not self.show_console:
                return
        
        context_id = self._get_context_id()
        formatted_msg = self._format_message(message, level)
        
        # Write to file
        if self.backend == 'file':
            f = self._get_or_create_file(context_id)
            f.write(formatted_msg + '\n')
            f.flush()

    def log(self, message: str, level: str = 'LOG'):
        """Write a log message"""
        if level not in self.LEVELS:
            level = 'LOG'
        
        self._queue_log(message, level)
        
        # Console output
        if self.show_console and self.LEVELS.index(level) >= self.LEVELS.index(self.show_level):
            self.original_stdout.write(self._format_message(message, level) + '\n')

    def debug(self, message: str):
        """Log debug message"""
        self.log(message, 'DEBUG')

    def info(self, message: str):
        """Log info message (alias for log)"""
        self.log(message, 'LOG')

    def warning(self, message: str):
        """Log warning message"""
        self.log(message, 'WARNING')

    def error(self, message: str):
        """Log error message"""
        self.log(message, 'ERROR')

    def _format_message(self, message: str, level: str = 'LOG') -> str:
        """Format message with timestamp and level"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        context_id = self._get_context_id()
        return f"[{timestamp}] [{level:8}] [{context_id}] {message}"

    def enable_capture(self, stdout: bool = True, stderr: bool = True):
        """Enable stdout/stderr capturing for current thread"""
        if stdout and sys.stdout != self:
            sys.stdout = StreamCapture(self.original_stdout, self)
        if stderr and sys.stderr != self:
            sys.stderr = StreamCapture(self.original_stderr, self)

    def disable_capture(self):
        """Disable stdout/stderr capturing"""
        if sys.stdout != self.original_stdout:
            sys.stdout = self.original_stdout
        if sys.stderr != self.original_stderr:
            sys.stderr = self.original_stderr

    @contextmanager
    def capture_context(self, stdout: bool = True, stderr: bool = True):
        """Context manager for temporary stdout/stderr capture"""
        self.enable_capture(stdout=stdout, stderr=stderr)
        try:
            yield
        finally:
            self.disable_capture()

    def flush(self):
        """Flush all open file handles"""
        with self.lock:
            for f in self.files.values():
                try:
                    f.flush()
                except:
                    pass

    def close(self):
        """Close all file handles"""
        with self.lock:
            for f in self.files.values():
                try:
                    f.close()
                except:
                    pass
            self.files.clear()
        
        self.disable_capture()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass

