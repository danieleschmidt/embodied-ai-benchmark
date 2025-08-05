"""Advanced caching system for the embodied AI benchmark."""

import time
import hashlib
import pickle
import json
from typing import Any, Dict, Optional, Callable, Union, Tuple
from pathlib import Path
from threading import Lock, RLock
from collections import OrderedDict
import logging
from datetime import datetime, timedelta
import weakref
import gc

logger = logging.getLogger(__name__)


class CacheStats:
    """Cache statistics tracking."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.entries = 0
        self.start_time = time.time()
        self._lock = Lock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_eviction(self, size_bytes: int):
        with self._lock:
            self.evictions += 1
            self.size_bytes -= size_bytes
            self.entries -= 1
    
    def record_insert(self, size_bytes: int):
        with self._lock:
            self.size_bytes += size_bytes
            self.entries += 1
    
    def get_hit_rate(self) -> float:
        total_requests = self.hits + self.misses
        return self.hits / max(1, total_requests)
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.get_hit_rate(),
            "size_bytes": self.size_bytes,
            "size_mb": self.size_bytes / (1024 * 1024),
            "entries": self.entries,
            "uptime_seconds": uptime
        }


class LRUCache:
    """High-performance LRU cache with size limits and TTL support."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 ttl_seconds: Optional[int] = None):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for entries in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        self._cache = OrderedDict()
        self._lock = RLock()
        self._stats = CacheStats()
        
        # Entry metadata: {key: (timestamp, size_bytes)}
        self._metadata = {}
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            else:
                return 1024  # Default estimate
    
    def _is_expired(self, key: str) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        timestamp = self._metadata.get(key, (0, 0))[0]
        return time.time() - timestamp > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired entries."""
        if self.ttl_seconds is None:
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, (timestamp, size_bytes) in self._metadata.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key, record_eviction=True)
    
    def _evict_lru(self):
        """Evict least recently used entries to free space."""
        while (len(self._cache) >= self.max_size or 
               self._stats.size_bytes > self.max_memory_bytes):
            if not self._cache:
                break
            
            # Remove oldest (LRU) entry
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key, record_eviction=True)
    
    def _remove_entry(self, key: str, record_eviction: bool = False):
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
        
        if key in self._metadata:
            _, size_bytes = self._metadata[key]
            del self._metadata[key]
            
            if record_eviction:
                self._stats.record_eviction(size_bytes)
            else:
                self._stats.size_bytes -= size_bytes
                self._stats.entries -= 1
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            # Clean expired entries periodically
            if len(self._cache) % 100 == 0:
                self._evict_expired()
            
            if key not in self._cache or self._is_expired(key):
                self._stats.record_miss()
                return default
            
            # Move to end (mark as recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            
            self._stats.record_hit()
            return value
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict if necessary
            self._evict_lru()
            
            # Insert new entry
            self._cache[key] = value
            self._metadata[key] = (time.time(), size_bytes)
            self._stats.record_insert(size_bytes)
    
    def invalidate(self, key: str):
        """Remove specific key from cache."""
        with self._lock:
            self._remove_entry(key)
    
    def clear(self):
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._stats.get_stats()


class AdaptiveCache:
    """Adaptive cache that adjusts size based on hit rate and memory pressure."""
    
    def __init__(self, 
                 initial_size: int = 500,
                 max_size: int = 2000,
                 target_hit_rate: float = 0.8,
                 adaptation_interval: int = 100):
        """Initialize adaptive cache.
        
        Args:
            initial_size: Initial cache size
            max_size: Maximum cache size
            target_hit_rate: Target hit rate for size adaptation
            adaptation_interval: Requests between adaptations
        """
        self.max_size = max_size
        self.target_hit_rate = target_hit_rate
        self.adaptation_interval = adaptation_interval
        
        self._cache = LRUCache(max_size=initial_size)
        self._request_count = 0
        self._last_adaptation = 0
        self._lock = Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with adaptive sizing."""
        with self._lock:
            self._request_count += 1
            
            # Adapt cache size periodically
            if (self._request_count - self._last_adaptation) >= self.adaptation_interval:
                self._adapt_size()
                self._last_adaptation = self._request_count
        
        return self._cache.get(key, default)
    
    def put(self, key: str, value: Any):
        """Put value in adaptive cache."""
        self._cache.put(key, value)
    
    def _adapt_size(self):
        """Adapt cache size based on performance."""
        stats = self._cache.get_stats()
        current_hit_rate = stats["hit_rate"]
        current_size = self._cache.max_size
        
        if current_hit_rate < self.target_hit_rate and current_size < self.max_size:
            # Increase cache size
            new_size = min(int(current_size * 1.2), self.max_size)
            self._cache.max_size = new_size
            logger.info(f"Increased cache size to {new_size} (hit rate: {current_hit_rate:.3f})")
        
        elif current_hit_rate > self.target_hit_rate * 1.1 and current_size > 100:
            # Decrease cache size if hit rate is very high
            new_size = max(int(current_size * 0.9), 100)
            self._cache.max_size = new_size
            logger.info(f"Decreased cache size to {new_size} (hit rate: {current_hit_rate:.3f})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive cache statistics."""
        stats = self._cache.get_stats()
        stats.update({
            "current_max_size": self._cache.max_size,
            "absolute_max_size": self.max_size,
            "target_hit_rate": self.target_hit_rate,
            "request_count": self._request_count
        })
        return stats


class PersistentCache:
    """Persistent cache with disk storage for expensive computations."""
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 memory_cache_size: int = 500,
                 compression: bool = True):
        """Initialize persistent cache.
        
        Args:
            cache_dir: Directory for persistent storage
            memory_cache_size: Size of in-memory cache
            compression: Whether to compress stored data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._memory_cache = LRUCache(max_size=memory_cache_size)
        self.compression = compression
        self._lock = Lock()
        
        # Index file for persistent entries
        self.index_file = self.cache_dir / "cache_index.json"
        self._persistent_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load persistent cache index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self):
        """Save persistent cache index."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._persistent_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cached entry."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.compression:
            import gzip
            data = gzip.compress(data)
        
        return data
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.compression:
            import gzip
            data = gzip.decompress(data)
        
        return pickle.loads(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache (memory first, then disk)."""
        # Check memory cache first
        value = self._memory_cache.get(key)
        if value is not None:
            return value
        
        # Check persistent cache
        with self._lock:
            if key in self._persistent_index:
                cache_path = self._get_cache_path(key)
                
                try:
                    if cache_path.exists():
                        with open(cache_path, 'rb') as f:
                            data = f.read()
                        
                        value = self._deserialize_value(data)
                        
                        # Add to memory cache
                        self._memory_cache.put(key, value)
                        
                        return value
                    else:
                        # Clean up stale index entry
                        del self._persistent_index[key]
                        self._save_index()
                
                except Exception as e:
                    logger.error(f"Failed to load cached value for {key}: {e}")
        
        return default
    
    def put(self, key: str, value: Any, persist: bool = True):
        """Put value in cache."""
        # Always add to memory cache
        self._memory_cache.put(key, value)
        
        # Optionally persist to disk
        if persist:
            with self._lock:
                try:
                    cache_path = self._get_cache_path(key)
                    data = self._serialize_value(value)
                    
                    with open(cache_path, 'wb') as f:
                        f.write(data)
                    
                    # Update index
                    self._persistent_index[key] = {
                        "timestamp": time.time(),
                        "size_bytes": len(data),
                        "file_path": str(cache_path)
                    }
                    self._save_index()
                
                except Exception as e:
                    logger.error(f"Failed to persist cached value for {key}: {e}")
    
    def invalidate(self, key: str):
        """Remove key from cache."""
        self._memory_cache.invalidate(key)
        
        with self._lock:
            if key in self._persistent_index:
                cache_path = self._get_cache_path(key)
                
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cached file for {key}: {e}")
                
                del self._persistent_index[key]
                self._save_index()
    
    def cleanup_old_entries(self, max_age_days: int = 7):
        """Remove old persistent cache entries."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        old_keys = []
        for key, metadata in self._persistent_index.items():
            if current_time - metadata["timestamp"] > max_age_seconds:
                old_keys.append(key)
        
        for key in old_keys:
            self.invalidate(key)
        
        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} old cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = self._memory_cache.get_stats()
        
        persistent_size = sum(
            metadata["size_bytes"] 
            for metadata in self._persistent_index.values()
        )
        
        return {
            "memory_cache": memory_stats,
            "persistent_entries": len(self._persistent_index),
            "persistent_size_bytes": persistent_size,
            "persistent_size_mb": persistent_size / (1024 * 1024)
        }


def cache_result(cache_instance: Optional[LRUCache] = None, 
                ttl_seconds: Optional[int] = None,
                key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        cache_instance: Cache instance to use (creates default if None)
        ttl_seconds: Time-to-live for cached results
        key_func: Function to generate cache key from arguments
    """
    if cache_instance is None:
        cache_instance = LRUCache(ttl_seconds=ttl_seconds)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)
            
            # Try to get from cache
            result = cache_instance.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)
            
            return result
        
        # Add cache management methods to wrapper
        wrapper.cache = cache_instance
        wrapper.cache_clear = cache_instance.clear
        wrapper.cache_stats = cache_instance.get_stats
        
        return wrapper
    
    return decorator


# Global cache instances
global_lru_cache = LRUCache(max_size=1000, max_memory_mb=50)
global_adaptive_cache = AdaptiveCache()
global_persistent_cache = PersistentCache()

# Cleanup old entries on module import
global_persistent_cache.cleanup_old_entries()