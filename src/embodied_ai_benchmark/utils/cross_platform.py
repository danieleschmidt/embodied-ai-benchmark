"""Cross-platform compatibility utilities for the embodied AI benchmark."""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import tempfile
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OperatingSystem(Enum):
    """Supported operating systems."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Supported architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    UNKNOWN = "unknown"


@dataclass
class SystemInfo:
    """System information container."""
    os: OperatingSystem
    architecture: Architecture
    python_version: str
    platform_details: str
    cpu_count: int
    memory_gb: float
    temp_dir: str
    path_separator: str
    line_separator: str
    supports_multiprocessing: bool
    supports_threading: bool
    max_path_length: int


class PlatformUtils:
    """Cross-platform utility functions."""
    
    @staticmethod
    def get_system_info() -> SystemInfo:
        """Get comprehensive system information.
        
        Returns:
            SystemInfo object with platform details
        """
        # Detect operating system
        system = platform.system().lower()
        if system == "windows":
            os_type = OperatingSystem.WINDOWS
        elif system == "linux":
            os_type = OperatingSystem.LINUX
        elif system == "darwin":
            os_type = OperatingSystem.MACOS
        else:
            os_type = OperatingSystem.UNKNOWN
        
        # Detect architecture
        machine = platform.machine().lower()
        if machine in ["x86_64", "amd64"]:
            arch = Architecture.X86_64
        elif machine in ["arm64", "aarch64"]:
            arch = Architecture.ARM64
        elif machine.startswith("arm"):
            arch = Architecture.ARM32
        else:
            arch = Architecture.UNKNOWN
        
        # Get memory info
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            memory_gb = 0.0
        
        # Path limitations
        max_path = 260 if os_type == OperatingSystem.WINDOWS else 4096
        
        return SystemInfo(
            os=os_type,
            architecture=arch,
            python_version=platform.python_version(),
            platform_details=platform.platform(),
            cpu_count=os.cpu_count() or 1,
            memory_gb=memory_gb,
            temp_dir=tempfile.gettempdir(),
            path_separator=os.sep,
            line_separator=os.linesep,
            supports_multiprocessing=hasattr(os, 'fork') or sys.platform == 'win32',
            supports_threading=threading.active_count() >= 0,
            max_path_length=max_path
        )
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """Normalize path for current platform.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized Path object
        """
        path_obj = Path(path)
        
        # Resolve relative paths
        if not path_obj.is_absolute():
            path_obj = path_obj.resolve()
        
        # Handle Windows drive letters
        if platform.system() == "Windows":
            # Ensure proper case for drive letters
            parts = path_obj.parts
            if parts and len(parts[0]) == 3 and parts[0][1] == ':':
                drive = parts[0][0].upper() + parts[0][1:]
                path_obj = Path(drive, *parts[1:])
        
        return path_obj
    
    @staticmethod
    def safe_path_join(*parts: str) -> Path:
        """Safely join path components across platforms.
        
        Args:
            *parts: Path components to join
            
        Returns:
            Joined path
        """
        # Filter out empty parts
        clean_parts = [part for part in parts if part]
        
        if not clean_parts:
            return Path()
        
        result = Path(clean_parts[0])
        for part in clean_parts[1:]:
            result = result / part
        
        return PlatformUtils.normalize_path(result)
    
    @staticmethod
    def get_executable_extension() -> str:
        """Get executable file extension for current platform.
        
        Returns:
            Executable extension ('.exe' on Windows, '' on Unix)
        """
        return '.exe' if platform.system() == "Windows" else ''
    
    @staticmethod
    def find_executable(name: str, search_paths: Optional[List[str]] = None) -> Optional[Path]:
        """Find executable in PATH or specified directories.
        
        Args:
            name: Executable name
            search_paths: Additional paths to search
            
        Returns:
            Path to executable if found
        """
        exe_ext = PlatformUtils.get_executable_extension()
        exe_name = name + exe_ext if not name.endswith(exe_ext) else name
        
        # Search in specified paths first
        if search_paths:
            for search_path in search_paths:
                exe_path = Path(search_path) / exe_name
                if exe_path.is_file() and os.access(exe_path, os.X_OK):
                    return exe_path
        
        # Search in PATH
        path_env = os.environ.get('PATH', '')
        for path_dir in path_env.split(os.pathsep):
            if path_dir:
                exe_path = Path(path_dir) / exe_name
                if exe_path.is_file() and os.access(exe_path, os.X_OK):
                    return exe_path
        
        return None
    
    @staticmethod
    def create_temp_directory(prefix: str = "benchmark_") -> Path:
        """Create temporary directory with proper permissions.
        
        Args:
            prefix: Directory name prefix
            
        Returns:
            Path to created temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        
        # Set appropriate permissions (readable/writable by owner only)
        if platform.system() != "Windows":
            os.chmod(temp_dir, 0o700)
        
        return temp_dir
    
    @staticmethod
    def safe_remove_directory(path: Union[str, Path], max_retries: int = 3) -> bool:
        """Safely remove directory tree with retry logic.
        
        Args:
            path: Directory to remove
            max_retries: Maximum retry attempts
            
        Returns:
            True if successful
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            return True
        
        for attempt in range(max_retries):
            try:
                if platform.system() == "Windows":
                    # Windows may need special handling for locked files
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, 0o777)
                        func(path)
                    
                    shutil.rmtree(path_obj, onerror=handle_remove_readonly)
                else:
                    shutil.rmtree(path_obj)
                
                return True
                
            except (OSError, PermissionError) as e:
                logger.warning(f"Failed to remove directory {path_obj} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        return False


class ProcessManager:
    """Cross-platform process management."""
    
    def __init__(self):
        """Initialize process manager."""
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.system_info = PlatformUtils.get_system_info()
    
    def start_process(self, 
                     command: List[str],
                     process_id: str,
                     working_dir: Optional[Union[str, Path]] = None,
                     env_vars: Optional[Dict[str, str]] = None,
                     capture_output: bool = True,
                     timeout: Optional[float] = None) -> Tuple[bool, str]:
        """Start a subprocess with cross-platform compatibility.
        
        Args:
            command: Command and arguments to execute
            process_id: Unique identifier for the process
            working_dir: Working directory for the process
            env_vars: Environment variables to set
            capture_output: Whether to capture stdout/stderr
            timeout: Process timeout in seconds
            
        Returns:
            Tuple of (success, message/error)
        """
        try:
            # Prepare environment
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            
            # Prepare working directory
            cwd = PlatformUtils.normalize_path(working_dir) if working_dir else None
            
            # Platform-specific process creation
            kwargs = {
                'args': command,
                'cwd': cwd,
                'env': env,
            }
            
            if capture_output:
                kwargs.update({
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.PIPE,
                    'text': True
                })
            
            # Windows-specific settings
            if self.system_info.os == OperatingSystem.WINDOWS:
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                # Unix-specific settings
                kwargs['start_new_session'] = True
            
            process = subprocess.Popen(**kwargs)
            self.active_processes[process_id] = process
            
            logger.info(f"Started process {process_id} with PID {process.pid}")
            return True, f"Process {process_id} started successfully"
            
        except Exception as e:
            logger.error(f"Failed to start process {process_id}: {e}")
            return False, str(e)
    
    def wait_for_process(self, 
                        process_id: str,
                        timeout: Optional[float] = None) -> Tuple[bool, int, str, str]:
        """Wait for process to complete.
        
        Args:
            process_id: Process identifier
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, return_code, stdout, stderr)
        """
        if process_id not in self.active_processes:
            return False, -1, "", "Process not found"
        
        process = self.active_processes[process_id]
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
            
            # Clean up
            del self.active_processes[process_id]
            
            success = return_code == 0
            return success, return_code, stdout or "", stderr or ""
            
        except subprocess.TimeoutExpired:
            self.terminate_process(process_id)
            return False, -1, "", "Process timed out"
        except Exception as e:
            logger.error(f"Error waiting for process {process_id}: {e}")
            return False, -1, "", str(e)
    
    def terminate_process(self, process_id: str, force: bool = False) -> bool:
        """Terminate a process gracefully or forcefully.
        
        Args:
            process_id: Process identifier
            force: Whether to force termination
            
        Returns:
            True if successful
        """
        if process_id not in self.active_processes:
            return False
        
        process = self.active_processes[process_id]
        
        try:
            if not force:
                # Graceful termination
                if self.system_info.os == OperatingSystem.WINDOWS:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    force = True
            
            if force:
                # Force termination
                if self.system_info.os == OperatingSystem.WINDOWS:
                    process.send_signal(signal.SIGTERM)
                else:
                    process.send_signal(signal.SIGKILL)
                
                process.wait(timeout=5.0)
            
            # Clean up
            del self.active_processes[process_id]
            logger.info(f"Terminated process {process_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate process {process_id}: {e}")
            return False
    
    def list_active_processes(self) -> List[str]:
        """Get list of active process IDs.
        
        Returns:
            List of active process identifiers
        """
        return list(self.active_processes.keys())
    
    def cleanup_all_processes(self):
        """Terminate all active processes."""
        for process_id in list(self.active_processes.keys()):
            self.terminate_process(process_id, force=True)


class ResourceMonitor:
    """Cross-platform resource monitoring."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.system_info = PlatformUtils.get_system_info()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._resource_data: List[Dict[str, Any]] = []
    
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current system resource usage.
        
        Returns:
            Dictionary with resource usage information
        """
        resources = {
            "timestamp": time.time(),
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_available_gb": 0.0,
            "disk_usage_percent": 0.0,
            "network_io": {"bytes_sent": 0, "bytes_recv": 0},
            "process_count": 0
        }
        
        try:
            import psutil
            
            # CPU usage
            resources["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            resources["memory_percent"] = memory.percent
            resources["memory_available_gb"] = memory.available / (1024**3)
            
            # Disk usage (for root/C: drive)
            if self.system_info.os == OperatingSystem.WINDOWS:
                disk = psutil.disk_usage('C:')
            else:
                disk = psutil.disk_usage('/')
            resources["disk_usage_percent"] = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                resources["network_io"]["bytes_sent"] = net_io.bytes_sent
                resources["network_io"]["bytes_recv"] = net_io.bytes_recv
            
            # Process count
            resources["process_count"] = len(psutil.pids())
            
        except ImportError:
            logger.warning("psutil not available, using basic resource monitoring")
        except Exception as e:
            logger.error(f"Error getting resource information: {e}")
        
        return resources
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._resource_data = []
        
        def monitor_loop():
            while self._monitoring:
                try:
                    resources = self.get_current_resources()
                    self._resource_data.append(resources)
                    
                    # Keep only last 1000 samples
                    if len(self._resource_data) > 1000:
                        self._resource_data = self._resource_data[-1000:]
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def get_resource_history(self) -> List[Dict[str, Any]]:
        """Get resource usage history.
        
        Returns:
            List of resource usage snapshots
        """
        return self._resource_data.copy()


# Global instances
system_info = PlatformUtils.get_system_info()
process_manager = ProcessManager()
resource_monitor = ResourceMonitor()


def get_platform_info() -> Dict[str, Any]:
    """Get platform information as dictionary.
    
    Returns:
        Platform information dictionary
    """
    info = system_info
    return {
        "operating_system": info.os.value,
        "architecture": info.architecture.value,
        "python_version": info.python_version,
        "platform_details": info.platform_details,
        "cpu_count": info.cpu_count,
        "memory_gb": info.memory_gb,
        "supports_multiprocessing": info.supports_multiprocessing,
        "supports_threading": info.supports_threading,
        "temp_directory": info.temp_dir,
        "path_separator": info.path_separator,
        "max_path_length": info.max_path_length
    }


def is_windows() -> bool:
    """Check if running on Windows."""
    return system_info.os == OperatingSystem.WINDOWS


def is_linux() -> bool:
    """Check if running on Linux."""
    return system_info.os == OperatingSystem.LINUX


def is_macos() -> bool:
    """Check if running on macOS."""
    return system_info.os == OperatingSystem.MACOS


def supports_multiprocessing() -> bool:
    """Check if multiprocessing is supported."""
    return system_info.supports_multiprocessing


class CrossPlatformManager:
    """Cross-platform management system with global optimization capabilities."""
    
    def __init__(self):
        """Initialize cross-platform manager."""
        self.system_info = system_info
        self.process_manager = process_manager
        self.resource_monitor = resource_monitor
        self.optimization_config = {
            "cpu_affinity_enabled": is_linux(),
            "memory_mapping_enabled": not is_windows(),
            "process_priority_management": True,
            "resource_monitoring_interval": 5.0,
            "auto_scaling_enabled": True
        }
        self._initialized = False
        self._optimization_thread: Optional[threading.Thread] = None
        self._optimization_active = False
    
    def initialize(self) -> bool:
        """Initialize the cross-platform management system.
        
        Returns:
            True if initialization successful
        """
        try:
            # Start resource monitoring
            self.resource_monitor.start_monitoring(
                interval=self.optimization_config["resource_monitoring_interval"]
            )
            
            # Initialize platform-specific optimizations
            if is_linux():
                self._setup_linux_optimizations()
            elif is_windows():
                self._setup_windows_optimizations()
            elif is_macos():
                self._setup_macos_optimizations()
            
            # Start optimization thread
            if self.optimization_config["auto_scaling_enabled"]:
                self._start_optimization_loop()
            
            self._initialized = True
            logger.info("CrossPlatformManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CrossPlatformManager: {e}")
            return False
    
    def _setup_linux_optimizations(self):
        """Set up Linux-specific optimizations."""
        try:
            # Enable CPU affinity if available
            if self.optimization_config["cpu_affinity_enabled"]:
                import os
                cpu_count = os.cpu_count() or 1
                # Set process affinity to use all CPUs
                try:
                    os.sched_setaffinity(0, set(range(cpu_count)))
                    logger.info(f"Set CPU affinity to {cpu_count} cores")
                except (OSError, AttributeError):
                    logger.warning("CPU affinity not available")
            
            # Configure memory management
            if self.optimization_config["memory_mapping_enabled"]:
                # Enable memory-mapped file optimizations
                logger.info("Linux memory optimizations enabled")
                
        except Exception as e:
            logger.warning(f"Linux optimization setup failed: {e}")
    
    def _setup_windows_optimizations(self):
        """Set up Windows-specific optimizations."""
        try:
            # Configure Windows-specific process priorities
            if self.optimization_config["process_priority_management"]:
                import os
                try:
                    # Set process priority to high
                    import psutil
                    process = psutil.Process(os.getpid())
                    process.nice(psutil.HIGH_PRIORITY_CLASS)
                    logger.info("Set Windows process priority to HIGH")
                except (ImportError, psutil.AccessDenied):
                    logger.warning("Process priority adjustment not available")
                    
        except Exception as e:
            logger.warning(f"Windows optimization setup failed: {e}")
    
    def _setup_macos_optimizations(self):
        """Set up macOS-specific optimizations."""
        try:
            # Configure macOS-specific optimizations
            if self.optimization_config["process_priority_management"]:
                import os
                try:
                    # Increase process priority
                    os.nice(-5)  # Higher priority
                    logger.info("Increased macOS process priority")
                except (OSError, PermissionError):
                    logger.warning("Process priority adjustment not available")
                    
        except Exception as e:
            logger.warning(f"macOS optimization setup failed: {e}")
    
    def _start_optimization_loop(self):
        """Start the optimization monitoring loop."""
        self._optimization_active = True
        
        def optimization_loop():
            while self._optimization_active:
                try:
                    # Get current resource usage
                    resources = self.resource_monitor.get_current_resources()
                    
                    # Apply dynamic optimizations based on resource usage
                    self._apply_dynamic_optimizations(resources)
                    
                    # Sleep for monitoring interval
                    time.sleep(self.optimization_config["resource_monitoring_interval"])
                    
                except Exception as e:
                    logger.error(f"Error in optimization loop: {e}")
                    time.sleep(5.0)
        
        self._optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self._optimization_thread.start()
        logger.info("Optimization monitoring loop started")
    
    def _apply_dynamic_optimizations(self, resources: Dict[str, Any]):
        """Apply dynamic optimizations based on current resource usage.
        
        Args:
            resources: Current resource usage information
        """
        try:
            # CPU-based optimizations
            if resources["cpu_percent"] > 80:
                # High CPU usage - reduce concurrent operations
                logger.debug("High CPU usage detected, reducing concurrency")
                
            elif resources["cpu_percent"] < 20:
                # Low CPU usage - increase concurrent operations
                logger.debug("Low CPU usage detected, increasing concurrency")
            
            # Memory-based optimizations
            if resources["memory_percent"] > 85:
                # High memory usage - trigger garbage collection
                import gc
                gc.collect()
                logger.debug("High memory usage, triggered garbage collection")
            
            # Disk-based optimizations
            if resources["disk_usage_percent"] > 90:
                logger.warning("High disk usage detected")
                
        except Exception as e:
            logger.error(f"Error applying dynamic optimizations: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status.
        
        Returns:
            Dictionary with optimization status information
        """
        return {
            "initialized": self._initialized,
            "optimization_active": self._optimization_active,
            "system_info": {
                "os": self.system_info.os.value,
                "architecture": self.system_info.architecture.value,
                "cpu_count": self.system_info.cpu_count,
                "memory_gb": self.system_info.memory_gb
            },
            "config": self.optimization_config,
            "active_processes": len(self.process_manager.active_processes),
            "resource_samples": len(self.resource_monitor.get_resource_history())
        }
    
    def optimize_for_task(self, task_type: str, expected_load: str = "medium") -> bool:
        """Optimize system configuration for specific task type.
        
        Args:
            task_type: Type of task (e.g., 'cpu_intensive', 'memory_intensive', 'io_intensive')
            expected_load: Expected load level ('low', 'medium', 'high')
            
        Returns:
            True if optimization applied successfully
        """
        try:
            if task_type == "cpu_intensive":
                # Optimize for CPU-bound tasks
                self.optimization_config["resource_monitoring_interval"] = 1.0
                if is_linux() and self.optimization_config["cpu_affinity_enabled"]:
                    # Spread across all CPUs
                    import os
                    cpu_count = os.cpu_count() or 1
                    os.sched_setaffinity(0, set(range(cpu_count)))
                    
            elif task_type == "memory_intensive":
                # Optimize for memory-bound tasks
                self.optimization_config["resource_monitoring_interval"] = 2.0
                # Pre-allocate memory pools if available
                import gc
                gc.set_threshold(700, 10, 10)  # More aggressive GC
                
            elif task_type == "io_intensive":
                # Optimize for I/O-bound tasks
                self.optimization_config["resource_monitoring_interval"] = 5.0
                # Enable asynchronous I/O optimizations
                
            logger.info(f"Optimized system for {task_type} tasks with {expected_load} load")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize for task type {task_type}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the cross-platform management system."""
        try:
            # Stop optimization loop
            self._optimization_active = False
            if self._optimization_thread:
                self._optimization_thread.join(timeout=5.0)
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            # Clean up processes
            self.process_manager.cleanup_all_processes()
            
            self._initialized = False
            logger.info("CrossPlatformManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during CrossPlatformManager shutdown: {e}")


# Global cross-platform manager instance
cross_platform_manager = CrossPlatformManager()