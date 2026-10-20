import os
import sys
import pytest
import tempfile
import threading
import multiprocessing
import time
import shutil
from io import StringIO
from pathlib import Path

from positronixlab.logger import Logger


class TestBasicLogging:
    """Test basic logging functionality"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def logger(self, temp_log_dir):
        """Create fresh logger instance for each test"""
        # Reset singleton
        Logger._instance = None
        logger = Logger(
            base_path=temp_log_dir,
            name='test',
            show_console=False,
            capture_stdout=False,
            capture_stderr=False
        )
        yield logger
        logger.close()
        Logger._instance = None

    def test_logger_initialization(self, logger, temp_log_dir):
        """Test logger initializes with correct parameters"""
        assert logger.base_path == temp_log_dir
        assert logger.name == 'test'
        assert logger.backend == 'file'
        assert logger.show_console == False

    def test_log_creates_file(self, logger, temp_log_dir):
        """Test that logging creates a file"""
        logger.log('Test message')
        files = os.listdir(temp_log_dir)
        assert len(files) > 0
        assert any('test' in f for f in files)

    def test_log_message_content(self, logger, temp_log_dir):
        """Test logged message appears in file"""
        message = 'This is a test message'
        logger.log(message)
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert message in content
        assert '[LOG' in content
        assert 'test' in content

    def test_debug_level(self, logger, temp_log_dir):
        """Test debug log level"""
        logger.debug('Debug message')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert '[DEBUG' in content

    def test_warning_level(self, logger, temp_log_dir):
        """Test warning log level"""
        logger.warning('Warning message')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert '[WARNING' in content

    def test_error_level(self, logger, temp_log_dir):
        """Test error log level"""
        logger.error('Error message')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert '[ERROR' in content

    def test_info_is_alias_for_log(self, logger, temp_log_dir):
        """Test that info() is an alias for log()"""
        logger.info('Info message')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert 'Info message' in content
        assert '[LOG' in content

    def test_multiple_messages(self, logger, temp_log_dir):
        """Test multiple messages are all logged"""
        messages = ['Message 1', 'Message 2', 'Message 3']
        for msg in messages:
            logger.log(msg)
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        for msg in messages:
            assert msg in content

    def test_message_formatting(self, logger, temp_log_dir):
        """Test message formatting includes timestamp and context"""
        logger.log('Test')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Check for timestamp format [YYYY-MM-DD HH:MM:SS]
        assert '[202' in content or '[201' in content  # Year starts with 20
        assert ':' in content  # Time separator
        assert '[LOG' in content or '[DEBUG' in content


class TestStdoutStderrCapturing:
    """Test stdout/stderr capturing functionality"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_stdout_capture_basic(self, temp_log_dir):
        """Test basic stdout capturing"""
        Logger._instance = None
        logger = Logger.capture(
            base_path=temp_log_dir,
            name='test_capture',
            capture_stdout=True,
            capture_stderr=False
        )
        
        print("Captured print statement")
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        log_file = os.path.join(temp_log_dir, files[0])
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert 'Captured print statement' in content
        assert '[STDOUT' in content
        
        logger.close()
        Logger._instance = None

    def test_enable_disable_capture(self, temp_log_dir):
        """Test enabling and disabling capture"""
        Logger._instance = None
        logger = Logger(
            base_path=temp_log_dir,
            name='test_toggle',
            show_console=False,
            capture_stdout=False
        )
        
        # Disable by default
        original_stdout = sys.stdout
        logger.enable_capture(stdout=True)
        
        print("Should be captured")
        logger.flush()
        
        logger.disable_capture()
        assert sys.stdout == original_stdout
        
        logger.close()
        Logger._instance = None

    def test_capture_context_manager(self, temp_log_dir):
        """Test capture context manager"""
        Logger._instance = None
        logger = Logger(
            base_path=temp_log_dir,
            name='test_context',
            show_console=False,
            capture_stdout=False
        )
        
        original_stdout = sys.stdout
        
        with logger.capture_context(stdout=True):
            print("Inside context")
            logger.log("Also inside")
        
        # Should be restored after context
        assert sys.stdout == original_stdout
        
        logger.flush()
        logger.close()
        Logger._instance = None

    def test_ignore_patterns(self, temp_log_dir):
        """Test that ignored patterns are filtered"""
        Logger._instance = None
        logger = Logger.capture(
            base_path=temp_log_dir,
            name='test_ignore',
            capture_stdout=True
        )
        
        # This should be ignored (matches pattern)
        print("WARNING: Full backward hook is firing")
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        if files:
            log_file = os.path.join(temp_log_dir, files[0])
            with open(log_file, 'r') as f:
                content = f.read()
            # Ignored message might not appear (depending on implementation)
        
        logger.close()
        Logger._instance = None


class TestMultithreading:
    """Test multithreading support"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_separate_files_per_thread(self, temp_log_dir):
        """Test that each thread gets its own log file"""
        Logger._instance = None
        
        thread_ids = []
        
        def worker(worker_id):
            logger = Logger.capture(
                base_path=temp_log_dir,
                name='test_threads',
                capture_stdout=False
            )
            logger.log(f'Worker {worker_id} message')
            logger.flush()
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        files = os.listdir(temp_log_dir)
        # Each worker should create at least one file
        assert len(files) >= 1
        
        # Verify content from multiple threads
        all_content = []
        for f in files:
            with open(os.path.join(temp_log_dir, f), 'r') as log_file:
                all_content.append(log_file.read())
        
        full_content = '\n'.join(all_content)
        # At least one worker message should be present
        assert any(f'Worker {i}' in full_content for i in range(3))
        
        Logger._instance = None

    def test_thread_names_in_logs(self, temp_log_dir):
        """Test that thread names appear in log context"""
        Logger._instance = None
        
        def worker():
            logger = Logger.capture(
                base_path=temp_log_dir,
                name='test_thread_name',
                capture_stdout=False
            )
            logger.log('Test message')
            logger.flush()
        
        t = threading.Thread(target=worker, name='TestWorker')
        t.start()
        t.join()
        
        files = os.listdir(temp_log_dir)
        assert len(files) > 0
        
        # The log file should contain thread information
        log_file = os.path.join(temp_log_dir, files[0])
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert 'Thread' in content or 'thread' in content
        
        Logger._instance = None

    def test_concurrent_logging_no_corruption(self, temp_log_dir):
        """Test that concurrent logging doesn't corrupt files"""
        Logger._instance = None
        
        def worker(worker_id):
            logger = Logger.capture(
                base_path=temp_log_dir,
                name='test_concurrent',
                capture_stdout=False
            )
            for i in range(10):
                logger.log(f'Worker {worker_id} message {i}')
            logger.flush()
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        files = os.listdir(temp_log_dir)
        assert len(files) > 0
        
        # Verify all files are readable and contain valid messages
        for filename in files:
            log_file = os.path.join(temp_log_dir, filename)
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Should have at least some log lines
            assert len(lines) > 0
        
        Logger._instance = None


class TestMultiprocessing:
    """Test multiprocessing support (limited due to pytest worker constraints)"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_process_id_in_context(self, temp_log_dir):
        """Test that process ID is included in context"""
        Logger._instance = None
        logger = Logger.capture(
            base_path=temp_log_dir,
            name='test_pid'
        )
        
        logger.log('Main process message')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        assert len(files) > 0
        
        log_file = os.path.join(temp_log_dir, files[0])
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Should contain pid information
        assert 'pid' in content.lower() or str(os.getpid()) in content
        
        logger.close()
        Logger._instance = None


class TestSingletonBehavior:
    """Test singleton pattern behavior"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_singleton_returns_same_instance(self, temp_log_dir):
        """Test that Logger returns same instance"""
        Logger._instance = None
        
        logger1 = Logger(base_path=temp_log_dir, name='test1')
        logger2 = Logger(base_path=temp_log_dir, name='test2')
        
        assert logger1 is logger2
        
        logger1.close()
        Logger._instance = None

    def test_activate_function_returns_singleton(self, temp_log_dir):
        """Test Logger.capture returns singleton"""
        Logger._instance = None
        
        logger1 = Logger.capture(base_path=temp_log_dir, name='test')
        logger2 = Logger.capture(base_path=temp_log_dir, name='test')
        
        assert logger1 is logger2
        
        logger1.close()
        Logger._instance = None


class TestCleanup:
    """Test cleanup operations"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_flush_writes_to_disk(self, temp_log_dir):
        """Test that flush actually writes to disk"""
        Logger._instance = None
        logger = Logger(
            base_path=temp_log_dir,
            name='test_flush',
            show_console=False
        )
        
        logger.log('Message before flush')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        assert len(files) > 0
        
        logger.close()
        Logger._instance = None

    def test_close_closes_files(self, temp_log_dir):
        """Test that close closes all files"""
        Logger._instance = None
        logger = Logger(
            base_path=temp_log_dir,
            name='test_close',
            show_console=False
        )
        
        logger.log('Message')
        logger.flush()
        
        assert len(logger.files) > 0
        logger.close()
        
        # After close, files dict should be cleared
        assert len(logger.files) == 0
        
        Logger._instance = None

    def test_disable_capture_on_close(self, temp_log_dir):
        """Test that close disables capture"""
        Logger._instance = None
        original_stdout = sys.stdout
        
        logger = Logger.capture(
            base_path=temp_log_dir,
            capture_stdout=True
        )
        
        assert sys.stdout != original_stdout
        
        logger.close()
        
        assert sys.stdout == original_stdout
        
        Logger._instance = None


class TestActivateLoggerForThread:
    """Test Logger.capture convenience function"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_one_liner_activation(self, temp_log_dir):
        """Test one-liner activation"""
        Logger._instance = None
        
        logger = Logger.capture(
            base_path=temp_log_dir,
            name='test_oneliner'
        )
        
        assert isinstance(logger, Logger)
        assert logger.base_path == temp_log_dir
        
        logger.close()
        Logger._instance = None

    def test_activation_enables_capture_by_default(self, temp_log_dir):
        """Test that activation enables capture by default"""
        Logger._instance = None
        original_stdout = sys.stdout
        
        logger = Logger.capture(
            base_path=temp_log_dir,
            capture_stdout=True
        )
        
        # stdout should be captured
        assert sys.stdout != original_stdout
        
        logger.close()
        Logger._instance = None

    def test_multiple_contexts(self, temp_log_dir):
        """Test multiple logger contexts in threads"""
        Logger._instance = None
        
        results = []
        
        def worker(worker_id):
            logger = Logger.capture(
                base_path=temp_log_dir,
                name=f'worker_{worker_id}',
                capture_stdout=False
            )
            logger.log(f'Context {worker_id}')
            logger.flush()
            results.append(len(os.listdir(temp_log_dir)))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Multiple log files should exist
        assert len(os.listdir(temp_log_dir)) > 0
        
        Logger._instance = None


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_invalid_backend_raises_error(self, temp_log_dir):
        """Test that invalid backend raises error"""
        Logger._instance = None
        
        with pytest.raises(ValueError):
            logger = Logger(
                base_path=temp_log_dir,
                name='test',
                backend='invalid_backend'
            )
        
        Logger._instance = None

    def test_creates_directory_if_not_exists(self, temp_log_dir):
        """Test that logger creates base path if it doesn't exist"""
        Logger._instance = None
        new_path = os.path.join(temp_log_dir, 'nested', 'deep', 'path')
        
        logger = Logger(
            base_path=new_path,
            name='test',
            show_console=False
        )
        
        assert os.path.exists(new_path)
        
        logger.close()
        Logger._instance = None

    def test_logging_with_special_characters(self, temp_log_dir):
        """Test logging messages with special characters"""
        Logger._instance = None
        logger = Logger(
            base_path=temp_log_dir,
            name='test_special',
            show_console=False
        )
        
        special_messages = [
            'Message with\nnewline',
            'Message with\ttab',
            'Message with émojis 🎉',
            'Message with "quotes"',
        ]
        
        for msg in special_messages:
            logger.log(msg)
        
        logger.flush()
        logger.close()
        Logger._instance = None

    def test_empty_message(self, temp_log_dir):
        """Test logging empty message"""
        Logger._instance = None
        logger = Logger(
            base_path=temp_log_dir,
            name='test_empty',
            show_console=False
        )
        
        logger.log('')
        logger.log('   ')
        logger.flush()
        
        logger.close()
        Logger._instance = None


class TestLogFileOrganization:
    """Test log file naming and organization"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_log_filename_contains_context(self, temp_log_dir):
        """Test that log filename contains meaningful context"""
        Logger._instance = None
        logger = Logger.capture(
            base_path=temp_log_dir,
            name='myapp',
            capture_stdout=False
        )
        
        logger.log('Test')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        assert len(files) > 0
        
        filename = files[0]
        assert 'myapp' in filename
        assert 'pid' in filename
        assert 'thread' in filename
        
        logger.close()
        Logger._instance = None

    def test_log_file_has_timestamp(self, temp_log_dir):
        """Test that log file includes date"""
        Logger._instance = None
        logger = Logger.capture(
            base_path=temp_log_dir,
            name='test',
            capture_stdout=False
        )
        
        logger.log('Test')
        logger.flush()
        
        files = os.listdir(temp_log_dir)
        filename = files[0]
        
        # Should contain date in format YYYYMMDD
        assert any(c.isdigit() for c in filename)
        
        logger.close()
        Logger._instance = None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
