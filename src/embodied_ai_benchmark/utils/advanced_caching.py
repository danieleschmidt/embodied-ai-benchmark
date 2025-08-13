"""Advanced caching system for embodied AI benchmarks."""

import asyncio
import hashlib
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import numpy as np
import logging

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    expiry_time: Optional[datetime] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expiry_time is None:
            return False
        return datetime.now() > self.expiry_time
    
    def touch(self):
        """Update access metadata."""
        self.accessed_at = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'expiry_time': self.expiry_time.isoformat() if self.expiry_time else None,
            'size_bytes': self.size_bytes,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache with advanced features."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._total_memory = 0
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cache entry or None if not found
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key)
                    self._misses += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                self._hits += 1
                return entry
            else:
                self._misses += 1
                return None
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Put cache entry.
        
        Args:
            key: Cache key
            entry: Cache entry
            
        Returns:
            True if entry was cached successfully
        """
        with self._lock:
            # Calculate entry size if not set
            if entry.size_bytes == 0:
                entry.size_bytes = self._calculate_size(entry.value)
            
            # Check if entry is too large
            if entry.size_bytes > self.max_memory_bytes:
                logger.warning(f"Cache entry {key} is too large ({entry.size_bytes} bytes)")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries to make space
            while (len(self._cache) >= self.max_size or 
                   self._total_memory + entry.size_bytes > self.max_memory_bytes) and self._cache:
                self._evict_lru()
            
            # Add new entry
            self._cache[key] = entry
            self._total_memory += entry.size_bytes
            
            logger.debug(f"Cached entry {key} ({entry.size_bytes} bytes)")
            return True
    
    def remove(self, key: str) -> bool:
        """Remove cache entry by key.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_memory = 0
    
    def _remove_entry(self, key: str):
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_memory -= entry.size_bytes
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._total_memory -= entry.size_bytes
            logger.debug(f"Evicted LRU entry {key}")
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback size estimation
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, (list, tuple)):
                return sum(self._calculate_size(item) for item in obj[:100])  # Limit for performance
            elif isinstance(obj, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in list(obj.items())[:100])
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                return 1024  # Default estimate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self._total_memory,
                'memory_usage_mb': self._total_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self._hits,
                'misses': self._misses,
                'total_requests': total_requests
            }


class PersistentCache:
    """Persistent cache using SQLite for storage."""
    
    def __init__(self, db_path: Union[str, Path], table_name: str = "cache_entries"):
        """Initialize persistent cache.
        
        Args:
            db_path: Path to SQLite database file
            table_name: Name of cache table
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._lock = threading.RLock()
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    expiry_time TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    tags TEXT DEFAULT '[]',
                    dependencies TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{{}}'
                )
            """)
            
            # Create indexes
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_expiry ON {self.table_name} (expiry_time)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_accessed ON {self.table_name} (accessed_at)")
            
            conn.commit()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from persistent storage.
        
        Args:
            key: Cache key
            
        Returns:
            Cache entry or None if not found
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"""
                        SELECT key, value, created_at, accessed_at, access_count, expiry_time, 
                               size_bytes, tags, dependencies, metadata
                        FROM {self.table_name} WHERE key = ?
                    """, (key,))
                    
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    
                    # Deserialize entry
                    entry = self._deserialize_entry(row)
                    
                    # Check expiration
                    if entry.is_expired():
                        self.remove(key)
                        return None
                    
                    # Update access info
                    entry.touch()
                    self._update_access_info(key, entry.accessed_at, entry.access_count)
                    
                    return entry
                    
            except Exception as e:
                logger.error(f"Error retrieving cache entry {key}: {e}")
                return None
    
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Store cache entry persistently.
        
        Args:
            key: Cache key
            entry: Cache entry
            
        Returns:
            True if entry was stored successfully
        """
        with self._lock:
            try:
                # Calculate size if not set
                if entry.size_bytes == 0:
                    entry.size_bytes = len(pickle.dumps(entry.value))
                
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    
                    # Serialize entry data
                    value_blob = pickle.dumps(entry.value)
                    tags_json = json.dumps(entry.tags)
                    deps_json = json.dumps(entry.dependencies)
                    metadata_json = json.dumps(entry.metadata)
                    expiry_str = entry.expiry_time.isoformat() if entry.expiry_time else None
                    
                    # Insert or replace entry
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO {self.table_name} 
                        (key, value, created_at, accessed_at, access_count, expiry_time, 
                         size_bytes, tags, dependencies, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, value_blob, entry.created_at.isoformat(), 
                        entry.accessed_at.isoformat(), entry.access_count, expiry_str,
                        entry.size_bytes, tags_json, deps_json, metadata_json
                    ))
                    
                    conn.commit()
                    return True
                    
            except Exception as e:
                logger.error(f"Error storing cache entry {key}: {e}")
                return False
    
    def remove(self, key: str) -> bool:
        """Remove cache entry from persistent storage.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                    conn.commit()
                    return cursor.rowcount > 0
                    
            except Exception as e:
                logger.error(f"Error removing cache entry {key}: {e}")
                return False
    
    def clear_expired(self) -> int:
        """Clear expired cache entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    cursor.execute(f"""
                        DELETE FROM {self.table_name} 
                        WHERE expiry_time IS NOT NULL AND expiry_time < ?
                    """, (now,))
                    conn.commit()
                    return cursor.rowcount
                    
            except Exception as e:
                logger.error(f"Error clearing expired entries: {e}")
                return 0
    
    def _update_access_info(self, key: str, accessed_at: datetime, access_count: int):
        """Update access information for entry."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    UPDATE {self.table_name} 
                    SET accessed_at = ?, access_count = ?
                    WHERE key = ?
                """, (accessed_at.isoformat(), access_count, key))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating access info for {key}: {e}")
    
    def _deserialize_entry(self, row: Tuple) -> CacheEntry:
        """Deserialize database row to CacheEntry."""
        (
            key, value_blob, created_at_str, accessed_at_str, access_count,
            expiry_time_str, size_bytes, tags_json, deps_json, metadata_json
        ) = row
        
        # Deserialize components
        value = pickle.loads(value_blob)
        created_at = datetime.fromisoformat(created_at_str)
        accessed_at = datetime.fromisoformat(accessed_at_str)
        expiry_time = datetime.fromisoformat(expiry_time_str) if expiry_time_str else None
        tags = json.loads(tags_json)
        dependencies = json.loads(deps_json)
        metadata = json.loads(metadata_json)
        
        return CacheEntry(
            key=key,
            value=value,
            created_at=created_at,
            accessed_at=accessed_at,
            access_count=access_count,
            expiry_time=expiry_time,
            size_bytes=size_bytes,
            tags=tags,
            dependencies=dependencies,
            metadata=metadata
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get persistent cache statistics."""
        with self._lock:
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    cursor = conn.cursor()
                    
                    # Count entries
                    cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    total_entries = cursor.fetchone()[0]
                    
                    # Count expired entries
                    now = datetime.now().isoformat()
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {self.table_name} 
                        WHERE expiry_time IS NOT NULL AND expiry_time < ?
                    """, (now,))
                    expired_entries = cursor.fetchone()[0]
                    
                    # Total size
                    cursor.execute(f"SELECT SUM(size_bytes) FROM {self.table_name}")
                    total_size = cursor.fetchone()[0] or 0
                    
                    # Database file size
                    db_file_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                    
                    return {
                        'total_entries': total_entries,
                        'expired_entries': expired_entries,
                        'active_entries': total_entries - expired_entries,
                        'total_size_bytes': total_size,
                        'total_size_mb': total_size / (1024 * 1024),
                        'db_file_size_bytes': db_file_size,
                        'db_file_size_mb': db_file_size / (1024 * 1024)
                    }
                    
            except Exception as e:
                logger.error(f"Error getting cache statistics: {e}")
                return {'error': str(e)}


class HierarchicalCache:
    """Multi-level cache with memory and persistent tiers."""
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        memory_cache_mb: float = 100.0,
        persistent_cache_path: Optional[Union[str, Path]] = None
    ):
        """Initialize hierarchical cache.
        
        Args:
            memory_cache_size: Maximum entries in memory cache
            memory_cache_mb: Maximum memory cache size in MB
            persistent_cache_path: Path for persistent cache database
        """
        self.memory_cache = LRUCache(memory_cache_size, memory_cache_mb)
        
        self.persistent_cache = None
        if persistent_cache_path:
            self.persistent_cache = PersistentCache(persistent_cache_path)
        
        self._lock = threading.RLock()
        
        # Statistics
        self._memory_hits = 0
        self._persistent_hits = 0
        self._total_misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from hierarchical cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            # Try memory cache first
            memory_entry = self.memory_cache.get(key)
            if memory_entry is not None:
                self._memory_hits += 1
                return memory_entry.value
            
            # Try persistent cache if available
            if self.persistent_cache:
                persistent_entry = self.persistent_cache.get(key)
                if persistent_entry is not None:
                    # Promote to memory cache
                    self.memory_cache.put(key, persistent_entry)
                    self._persistent_hits += 1
                    return persistent_entry.value
            
            self._total_misses += 1
            return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = True
    ) -> bool:
        """Put value into hierarchical cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            tags: Tags for cache entry
            dependencies: Dependencies for cache entry
            metadata: Additional metadata
            persist: Whether to store in persistent cache
            
        Returns:
            True if value was cached successfully
        """
        with self._lock:
            # Create cache entry
            now = datetime.now()
            expiry_time = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                expiry_time=expiry_time,
                tags=tags or [],
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            # Store in memory cache
            memory_success = self.memory_cache.put(key, entry)
            
            # Store in persistent cache if requested and available
            persistent_success = True
            if persist and self.persistent_cache:
                persistent_success = self.persistent_cache.put(key, entry)
            
            return memory_success or persistent_success
    
    def remove(self, key: str) -> bool:
        """Remove entry from both cache levels.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed from any level
        """
        with self._lock:
            memory_removed = self.memory_cache.remove(key)
            persistent_removed = False
            
            if self.persistent_cache:
                persistent_removed = self.persistent_cache.remove(key)
            
            return memory_removed or persistent_removed
    
    def clear_expired(self) -> Dict[str, int]:
        """Clear expired entries from both cache levels.
        
        Returns:
            Dictionary with counts of removed entries per level
        """
        with self._lock:
            persistent_cleared = 0
            if self.persistent_cache:
                persistent_cleared = self.persistent_cache.clear_expired()
            
            # Memory cache expiration is handled on access
            return {
                'memory_cleared': 0,  # Cleared on access
                'persistent_cleared': persistent_cleared
            }
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags.
        
        Args:
            tags: Tags to match for invalidation
            
        Returns:
            Number of entries invalidated
        """
        # This would require additional indexing in the persistent cache
        # For now, return 0 as placeholder
        logger.warning("Tag-based invalidation not yet implemented")
        return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            memory_stats = self.memory_cache.get_statistics()
            persistent_stats = self.persistent_cache.get_statistics() if self.persistent_cache else {}
            
            total_requests = self._memory_hits + self._persistent_hits + self._total_misses
            overall_hit_rate = ((self._memory_hits + self._persistent_hits) / 
                               total_requests if total_requests > 0 else 0)
            
            return {
                'memory_cache': memory_stats,
                'persistent_cache': persistent_stats,
                'hierarchical_stats': {
                    'memory_hits': self._memory_hits,
                    'persistent_hits': self._persistent_hits,
                    'total_misses': self._total_misses,
                    'total_requests': total_requests,
                    'overall_hit_rate': overall_hit_rate,
                    'memory_hit_rate': self._memory_hits / total_requests if total_requests > 0 else 0
                }
            }


def cache_key_from_function_args(func: Callable, *args, **kwargs) -> str:
    """Generate cache key from function name and arguments.
    
    Args:
        func: Function object
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Create a deterministic representation of the arguments
    arg_repr = []
    
    # Add function name
    func_name = f"{func.__module__}.{func.__name__}"
    arg_repr.append(func_name)
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            arg_repr.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            # Hash complex structures
            arg_hash = hashlib.md5(str(arg).encode()).hexdigest()[:8]
            arg_repr.append(f"{type(arg).__name__}:{arg_hash}")
        elif isinstance(arg, dict):
            # Sort dict keys for deterministic hash
            sorted_items = sorted(arg.items())
            arg_hash = hashlib.md5(str(sorted_items).encode()).hexdigest()[:8]
            arg_repr.append(f"dict:{arg_hash}")
        elif isinstance(arg, np.ndarray):
            # Hash array shape and dtype
            array_info = f"{arg.shape}_{arg.dtype}"
            arg_hash = hashlib.md5(array_info.encode()).hexdigest()[:8]
            arg_repr.append(f"array:{arg_hash}")
        else:
            # Generic object hash
            obj_hash = hashlib.md5(str(type(arg)).encode()).hexdigest()[:8]
            arg_repr.append(f"{type(arg).__name__}:{obj_hash}")
    
    # Add keyword arguments
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = str(sorted_kwargs)
        kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()[:8]
        arg_repr.append(f"kwargs:{kwargs_hash}")
    
    # Combine into cache key
    cache_key = "|".join(arg_repr)
    
    # Hash if too long
    if len(cache_key) > 200:
        cache_key = hashlib.sha256(cache_key.encode()).hexdigest()
    
    return cache_key


def cached_function(
    cache: HierarchicalCache,
    ttl_seconds: Optional[float] = None,
    tags: Optional[List[str]] = None,
    persist: bool = True
):
    """Decorator for caching function results.
    
    Args:
        cache: Cache instance to use
        ttl_seconds: Time-to-live for cached results
        tags: Tags for cache entries
        persist: Whether to persist cache entries
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_key_from_function_args(func, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__} with key {cache_key[:20]}...")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache result
            metadata = {
                'function_name': func.__name__,
                'execution_time': execution_time,
                'cached_at': datetime.now().isoformat()
            }
            
            cache.put(
                cache_key, result,
                ttl_seconds=ttl_seconds,
                tags=tags,
                metadata=metadata,
                persist=persist
            )
            
            return result
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._original_function = func
        wrapper._cache = cache
        
        return wrapper
    
    return decorator


# Global cache instance for convenience
_global_cache = None


def get_global_cache() -> HierarchicalCache:
    """Get or create global cache instance.
    
    Returns:
        Global cache instance
    """
    global _global_cache
    if _global_cache is None:
        cache_dir = Path.home() / ".embodied_ai_benchmark" / "cache"
        cache_db_path = cache_dir / "benchmark_cache.db"
        
        _global_cache = HierarchicalCache(
            memory_cache_size=2000,
            memory_cache_mb=200.0,
            persistent_cache_path=cache_db_path
        )
    
    return _global_cache


def cached(
    ttl_seconds: Optional[float] = 3600.0,  # 1 hour default
    tags: Optional[List[str]] = None,
    persist: bool = True
):
    """Convenient decorator using global cache.
    
    Args:
        ttl_seconds: Time-to-live for cached results
        tags: Tags for cache entries
        persist: Whether to persist cache entries
        
    Returns:
        Decorated function
    """
    return cached_function(
        get_global_cache(),
        ttl_seconds=ttl_seconds,
        tags=tags,
        persist=persist
    )
