"""Advanced performance optimization engine with intelligent caching and auto-scaling."""

import time
import json
import threading
import hashlib
import pickle
import gc
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict, OrderedDict
from functools import wraps
import concurrent.futures
import sqlite3
import os


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    function_name: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    cache_hit: bool
    timestamp: datetime
    input_hash: str
    output_size: int


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int]
    tags: List[str]


class AdvancedCacheEngine:
    """Multi-level caching engine with intelligent eviction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize cache engine.
        
        Args:
            config: Cache configuration
        """
        self.config = config or {}
        
        # Cache storage layers
        self.memory_cache = OrderedDict()  # L1: Memory cache
        self.disk_cache_path = self.config.get("disk_cache_path", "/tmp/ai_benchmark_cache")
        
        # Cache configuration
        self.max_memory_size = self.config.get("max_memory_mb", 500) * 1024 * 1024  # bytes
        self.max_disk_size = self.config.get("max_disk_mb", 2000) * 1024 * 1024  # bytes
        self.default_ttl = self.config.get("default_ttl_seconds", 3600)  # 1 hour
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.current_memory_usage = 0
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Initialize disk cache
        self._init_disk_cache()
        self._start_cleanup_thread()
    
    def _init_disk_cache(self):
        """Initialize disk-based cache."""
        os.makedirs(self.disk_cache_path, exist_ok=True)
        
        # Create metadata database
        self.db_path = os.path.join(self.disk_cache_path, "cache_metadata.db")
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                filename TEXT,
                timestamp REAL,
                access_count INTEGER,
                size_bytes INTEGER,
                ttl_seconds INTEGER,
                tags TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="CacheCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_expired_entries()
                self._enforce_size_limits()
                self._stop_cleanup.wait(300)  # Cleanup every 5 minutes
            except Exception as e:
                # Log error but continue
                pass
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (value, was_hit)
        """
        with self._lock:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    del self.memory_cache[key]
                    self.current_memory_usage -= entry.size_bytes
                else:
                    # Update access and move to end (LRU)
                    entry.access_count += 1
                    self.memory_cache.move_to_end(key)
                    self.cache_hits += 1
                    return entry.value, True
            
            # Try disk cache
            disk_value = self._get_from_disk(key)
            if disk_value is not None:
                self.cache_hits += 1
                
                # Promote to memory cache if frequently accessed
                self._maybe_promote_to_memory(key, disk_value)
                return disk_value, True
            
            self.cache_misses += 1
            return None, False
    
    def set(self, 
           key: str, 
           value: Any, 
           ttl_seconds: Optional[int] = None,
           tags: List[str] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Cache tags for grouping
            
        Returns:
            Whether value was successfully cached
        """
        try:
            with self._lock:
                # Serialize value to estimate size
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
                
                # Create cache entry
                entry = CacheEntry(
                    value=value,
                    timestamp=datetime.now(timezone.utc),
                    access_count=1,
                    size_bytes=size_bytes,
                    ttl_seconds=ttl_seconds or self.default_ttl,
                    tags=tags or []
                )
                
                # Decide caching strategy based on size
                if size_bytes < 1024 * 1024:  # < 1MB: memory cache
                    self._set_in_memory(key, entry)
                else:  # >= 1MB: disk cache
                    self._set_on_disk(key, entry, serialized)
                
                return True
                
        except Exception as e:
            return False
    
    def _set_in_memory(self, key: str, entry: CacheEntry):
        """Set entry in memory cache."""
        # Check if we need to evict
        while (self.current_memory_usage + entry.size_bytes > self.max_memory_size 
               and self.memory_cache):
            # Evict least recently used
            oldest_key = next(iter(self.memory_cache))
            oldest_entry = self.memory_cache.pop(oldest_key)
            self.current_memory_usage -= oldest_entry.size_bytes
            
            # Move to disk cache if valuable
            if oldest_entry.access_count > 1:
                try:
                    serialized = pickle.dumps(oldest_entry.value)
                    self._set_on_disk(oldest_key, oldest_entry, serialized)
                except:
                    pass  # Failed to serialize, just discard
        
        # Add new entry
        self.memory_cache[key] = entry
        self.current_memory_usage += entry.size_bytes
    
    def _set_on_disk(self, key: str, entry: CacheEntry, serialized: bytes):
        """Set entry in disk cache."""
        try:
            # Generate filename
            filename = hashlib.md5(key.encode()).hexdigest() + ".cache"
            filepath = os.path.join(self.disk_cache_path, filename)
            
            # Write data
            with open(filepath, 'wb') as f:
                f.write(serialized)
            
            # Update metadata
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO cache_metadata 
                (key, filename, timestamp, access_count, size_bytes, ttl_seconds, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                key, filename, entry.timestamp.timestamp(),
                entry.access_count, entry.size_bytes,
                entry.ttl_seconds, json.dumps(entry.tags)
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            # Failed to write to disk
            pass
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT filename, timestamp, ttl_seconds FROM cache_metadata WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            filename, timestamp, ttl_seconds = row
            
            # Check TTL
            if ttl_seconds and time.time() - timestamp > ttl_seconds:
                self._remove_from_disk(key)
                return None
            
            # Read data
            filepath = os.path.join(self.disk_cache_path, filename)
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'rb') as f:
                data = f.read()
            
            value = pickle.loads(data)
            
            # Update access count
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE cache_metadata SET access_count = access_count + 1 WHERE key = ?",
                (key,)
            )
            conn.commit()
            conn.close()
            
            return value
            
        except Exception as e:
            return None
    
    def _maybe_promote_to_memory(self, key: str, value: Any):
        """Promote frequently accessed disk entries to memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT access_count, size_bytes FROM cache_metadata WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0] > 3 and row[1] < 512 * 1024:  # > 3 accesses, < 512KB
                entry = CacheEntry(
                    value=value,
                    timestamp=datetime.now(timezone.utc),
                    access_count=row[0],
                    size_bytes=row[1],
                    ttl_seconds=self.default_ttl,
                    tags=[]
                )
                self._set_in_memory(key, entry)
                
        except Exception:
            pass
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if not entry.ttl_seconds:
            return False
        
        age = (datetime.now(timezone.utc) - entry.timestamp).total_seconds()
        return age > entry.ttl_seconds
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        with self._lock:
            # Clean memory cache
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                entry = self.memory_cache.pop(key)
                self.current_memory_usage -= entry.size_bytes
        
        # Clean disk cache
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT key, filename FROM cache_metadata 
                WHERE ttl_seconds > 0 AND ? - timestamp > ttl_seconds
            """, (time.time(),))
            
            expired_entries = cursor.fetchall()
            
            for key, filename in expired_entries:
                self._remove_from_disk(key)
            
            conn.close()
            
        except Exception:
            pass
    
    def _enforce_size_limits(self):
        """Enforce cache size limits."""
        # Memory cache is handled during insertion
        
        # Disk cache cleanup
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_metadata")
            total_size = cursor.fetchone()[0] or 0
            
            if total_size > self.max_disk_size:
                # Remove least recently used entries
                cursor = conn.execute("""
                    SELECT key FROM cache_metadata 
                    ORDER BY access_count ASC, timestamp ASC
                    LIMIT ?
                """, (max(1, len(self.memory_cache) // 4),))
                
                keys_to_remove = [row[0] for row in cursor.fetchall()]
                
                for key in keys_to_remove:
                    self._remove_from_disk(key)
            
            conn.close()
            
        except Exception:
            pass
    
    def _remove_from_disk(self, key: str):
        """Remove entry from disk cache."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT filename FROM cache_metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            
            if row:
                filename = row[0]
                filepath = os.path.join(self.disk_cache_path, filename)
                
                # Remove file
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # Remove metadata
                conn.execute("DELETE FROM cache_metadata WHERE key = ?", (key,))
                conn.commit()
            
            conn.close()
            
        except Exception:
            pass
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags."""
        with self._lock:
            # Memory cache
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self.memory_cache.pop(key)
                self.current_memory_usage -= entry.size_bytes
        
        # Disk cache
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT key, tags FROM cache_metadata")
            
            for key, tags_json in cursor.fetchall():
                try:
                    entry_tags = json.loads(tags_json)
                    if any(tag in entry_tags for tag in tags):
                        self._remove_from_disk(key)
                except:
                    pass
            
            conn.close()
            
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        
        return {
            "hit_rate": hit_rate,
            "total_hits": self.cache_hits,
            "total_misses": self.cache_misses,
            "memory_entries": len(self.memory_cache),
            "memory_usage_mb": self.current_memory_usage / (1024 * 1024),
            "memory_limit_mb": self.max_memory_size / (1024 * 1024)
        }


class PerformanceOptimizer:
    """Performance optimization engine with auto-tuning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance optimizer.
        
        Args:
            config: Optimizer configuration
        """
        self.config = config or {}
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.optimization_rules = {}
        
        # Resource monitoring
        self.cpu_threshold = self.config.get("cpu_threshold", 80)
        self.memory_threshold = self.config.get("memory_threshold", 85)
        
        # Optimization strategies
        self.batch_size = self.config.get("default_batch_size", 32)
        self.max_workers = self.config.get("max_workers", 4)
        
        # Caching
        self.cache = AdvancedCacheEngine(self.config.get("cache", {}))
        
        # Threading
        self._lock = threading.RLock()
    
    def register_optimization_rule(self, 
                                 function_name: str,
                                 rule: Callable[[List[PerformanceMetrics]], Dict[str, Any]]):
        """Register optimization rule for function.
        
        Args:
            function_name: Function to optimize
            rule: Optimization rule function
        """
        with self._lock:
            self.optimization_rules[function_name] = rule
    
    def track_performance(self, metrics: PerformanceMetrics):
        """Track performance metrics."""
        with self._lock:
            self.performance_history[metrics.function_name].append(metrics)
            
            # Limit history size
            max_history = self.config.get("max_history_per_function", 1000)
            if len(self.performance_history[metrics.function_name]) > max_history:
                self.performance_history[metrics.function_name] = (
                    self.performance_history[metrics.function_name][-max_history:]
                )
            
            # Auto-optimize if enough data
            if len(self.performance_history[metrics.function_name]) > 10:
                self._auto_optimize(metrics.function_name)
    
    def _auto_optimize(self, function_name: str):
        """Automatically optimize function based on performance history."""
        if function_name not in self.optimization_rules:
            return
        
        history = self.performance_history[function_name]
        rule = self.optimization_rules[function_name]
        
        try:
            optimizations = rule(history)
            
            # Apply optimizations
            if optimizations:
                self._apply_optimizations(function_name, optimizations)
                
        except Exception as e:
            # Log optimization failure
            pass
    
    def _apply_optimizations(self, function_name: str, optimizations: Dict[str, Any]):
        """Apply optimization suggestions."""
        # This would be implemented per optimization type
        # For now, just track the suggestions
        pass
    
    def batch_process(self, 
                     items: List[Any],
                     processor: Callable[[Any], Any],
                     batch_size: Optional[int] = None) -> List[Any]:
        """Process items in optimized batches.
        
        Args:
            items: Items to process
            processor: Processing function
            batch_size: Override batch size
            
        Returns:
            Processed results
        """
        batch_size = batch_size or self._get_optimal_batch_size(len(items))
        results = []
        
        # Process in parallel batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                future = executor.submit(self._process_batch, batch, processor)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    # Handle batch failure
                    pass
        
        return results
    
    def _process_batch(self, batch: List[Any], processor: Callable[[Any], Any]) -> List[Any]:
        """Process a single batch."""
        return [processor(item) for item in batch]
    
    def _get_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on system resources."""
        # Simple heuristic - would be more sophisticated in practice
        if total_items < 100:
            return min(total_items, 16)
        elif total_items < 1000:
            return 32
        else:
            return 64
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_stats": self.cache.get_stats(),
            "function_performance": {},
            "optimization_suggestions": []
        }
        
        for func_name, history in self.performance_history.items():
            if not history:
                continue
            
            execution_times = [m.execution_time for m in history]
            memory_usage = [m.memory_usage for m in history]
            cache_hit_rate = sum(1 for m in history if m.cache_hit) / len(history)
            
            report["function_performance"][func_name] = {
                "call_count": len(history),
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "avg_memory_usage": sum(memory_usage) / len(memory_usage),
                "cache_hit_rate": cache_hit_rate,
                "last_called": history[-1].timestamp.isoformat()
            }
            
            # Generate suggestions
            avg_time = report["function_performance"][func_name]["avg_execution_time"]
            if avg_time > 1.0:  # > 1 second
                report["optimization_suggestions"].append({
                    "function": func_name,
                    "issue": "slow_execution",
                    "suggestion": "Consider optimizing algorithm or adding caching",
                    "avg_time": avg_time
                })
            
            if cache_hit_rate < 0.5:  # < 50% hit rate
                report["optimization_suggestions"].append({
                    "function": func_name,
                    "issue": "low_cache_hit_rate",
                    "suggestion": "Review caching strategy or increase cache size",
                    "hit_rate": cache_hit_rate
                })
        
        return report


def performance_monitor(optimizer: PerformanceOptimizer = None,
                       cache_key_func: Callable = None,
                       cache_ttl: int = 3600):
    """Decorator for automatic performance monitoring and caching.
    
    Args:
        optimizer: Performance optimizer instance
        cache_key_func: Function to generate cache key
        cache_ttl: Cache time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0  # Would use psutil in full environment
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Simple hash of arguments
                arg_str = f"{args}_{kwargs}"
                cache_key = f"{func.__name__}_{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Try cache first
            if optimizer:
                cached_result, cache_hit = optimizer.cache.get(cache_key)
                if cache_hit:
                    # Record cache hit metrics
                    metrics = PerformanceMetrics(
                        function_name=func.__name__,
                        execution_time=0.001,  # Minimal cache retrieval time
                        memory_usage=0,
                        cpu_usage=0,
                        cache_hit=True,
                        timestamp=datetime.now(timezone.utc),
                        input_hash=cache_key,
                        output_size=len(str(cached_result)) if cached_result else 0
                    )
                    optimizer.track_performance(metrics)
                    return cached_result
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                end_memory = 0  # Would use psutil in full environment
                memory_usage = max(0, end_memory - start_memory)
                
                # Cache result
                if optimizer:
                    optimizer.cache.set(cache_key, result, cache_ttl)
                
                # Record metrics
                if optimizer:
                    metrics = PerformanceMetrics(
                        function_name=func.__name__,
                        execution_time=execution_time,
                        memory_usage=memory_usage,
                        cpu_usage=0,  # Would calculate from psutil
                        cache_hit=False,
                        timestamp=datetime.now(timezone.utc),
                        input_hash=cache_key,
                        output_size=len(str(result)) if result else 0
                    )
                    optimizer.track_performance(metrics)
                
                return result
                
            except Exception as e:
                # Record failed execution
                execution_time = time.time() - start_time
                
                if optimizer:
                    metrics = PerformanceMetrics(
                        function_name=func.__name__,
                        execution_time=execution_time,
                        memory_usage=0,
                        cpu_usage=0,
                        cache_hit=False,
                        timestamp=datetime.now(timezone.utc),
                        input_hash=cache_key,
                        output_size=0
                    )
                    optimizer.track_performance(metrics)
                
                raise e
        
        return wrapper
    return decorator


# Global optimizer instance
_global_optimizer = None

def get_global_optimizer() -> PerformanceOptimizer:
    """Get or create global optimizer."""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    
    return _global_optimizer