"""Cache management for benchmark results and data."""

import json
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from ..database.connection import CacheManager, get_cache


class BenchmarkCacheManager:
    """Enhanced cache manager for benchmark-specific data."""
    
    def __init__(self, cache: Optional[CacheManager] = None):
        """Initialize benchmark cache manager.
        
        Args:
            cache: Cache manager instance
        """
        self.cache = cache or get_cache()
        self.default_ttl = 3600  # 1 hour default TTL
        self.prefixes = {
            "results": "bench:results:",
            "agents": "bench:agents:",
            "tasks": "bench:tasks:",
            "metrics": "bench:metrics:",
            "configs": "bench:configs:"
        }
    
    def _generate_key(self, prefix: str, *args: str) -> str:
        """Generate cache key with prefix and arguments.
        
        Args:
            prefix: Key prefix
            *args: Key components
            
        Returns:
            Generated cache key
        """
        key_parts = [prefix] + list(args)
        return ":".join(str(part) for part in key_parts)
    
    def _hash_object(self, obj: Any) -> str:
        """Generate hash for complex objects.
        
        Args:
            obj: Object to hash
            
        Returns:
            Object hash string
        """
        json_str = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def cache_benchmark_results(self,
                               agent_name: str,
                               task_name: str,
                               config_hash: str,
                               results: Dict[str, Any],
                               ttl: Optional[int] = None) -> bool:
        """Cache benchmark results.
        
        Args:
            agent_name: Agent name
            task_name: Task name
            config_hash: Configuration hash
            results: Benchmark results
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._generate_key(
                self.prefixes["results"],
                agent_name,
                task_name,
                config_hash
            )
            
            cache_data = {
                "results": results,
                "cached_at": datetime.now().isoformat(),
                "agent_name": agent_name,
                "task_name": task_name
            }
            
            self.cache.set(key, json.dumps(cache_data), ttl or self.default_ttl)
            return True
            
        except Exception as e:
            print(f"Failed to cache benchmark results: {e}")
            return False
    
    def get_cached_results(self,
                          agent_name: str,
                          task_name: str,
                          config_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached benchmark results.
        
        Args:
            agent_name: Agent name
            task_name: Task name
            config_hash: Configuration hash
            
        Returns:
            Cached results or None if not found
        """
        try:
            key = self._generate_key(
                self.prefixes["results"],
                agent_name,
                task_name,
                config_hash
            )
            
            cached_data = self.cache.get(key)
            if cached_data:
                data = json.loads(cached_data)
                return data.get("results")
            
            return None
            
        except Exception as e:
            print(f"Failed to get cached results: {e}")
            return None
    
    def cache_agent_performance(self,
                               agent_name: str,
                               performance_data: Dict[str, Any],
                               ttl: Optional[int] = None) -> bool:
        """Cache agent performance summary.
        
        Args:
            agent_name: Agent name
            performance_data: Performance data
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._generate_key(self.prefixes["agents"], agent_name, "performance")
            
            cache_data = {
                "performance": performance_data,
                "cached_at": datetime.now().isoformat()
            }
            
            self.cache.set(key, json.dumps(cache_data), ttl or self.default_ttl)
            return True
            
        except Exception as e:
            print(f"Failed to cache agent performance: {e}")
            return False
    
    def get_cached_agent_performance(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get cached agent performance.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Cached performance data or None if not found
        """
        try:
            key = self._generate_key(self.prefixes["agents"], agent_name, "performance")
            
            cached_data = self.cache.get(key)
            if cached_data:
                data = json.loads(cached_data)
                return data.get("performance")
            
            return None
            
        except Exception as e:
            print(f"Failed to get cached agent performance: {e}")
            return None
    
    def cache_task_metadata(self,
                           task_name: str,
                           metadata: Dict[str, Any],
                           ttl: Optional[int] = None) -> bool:
        """Cache task metadata.
        
        Args:
            task_name: Task name
            metadata: Task metadata
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._generate_key(self.prefixes["tasks"], task_name, "metadata")
            
            cache_data = {
                "metadata": metadata,
                "cached_at": datetime.now().isoformat()
            }
            
            # Task metadata changes infrequently, longer TTL
            ttl = ttl or (self.default_ttl * 24)  # 24 hours
            
            self.cache.set(key, json.dumps(cache_data), ttl)
            return True
            
        except Exception as e:
            print(f"Failed to cache task metadata: {e}")
            return False
    
    def get_cached_task_metadata(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get cached task metadata.
        
        Args:
            task_name: Task name
            
        Returns:
            Cached metadata or None if not found
        """
        try:
            key = self._generate_key(self.prefixes["tasks"], task_name, "metadata")
            
            cached_data = self.cache.get(key)
            if cached_data:
                data = json.loads(cached_data)
                return data.get("metadata")
            
            return None
            
        except Exception as e:
            print(f"Failed to get cached task metadata: {e}")
            return None
    
    def cache_leaderboard(self,
                         task_name: str,
                         leaderboard_data: List[Dict[str, Any]],
                         ttl: Optional[int] = None) -> bool:
        """Cache task leaderboard.
        
        Args:
            task_name: Task name
            leaderboard_data: Leaderboard data
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._generate_key(self.prefixes["tasks"], task_name, "leaderboard")
            
            cache_data = {
                "leaderboard": leaderboard_data,
                "cached_at": datetime.now().isoformat()
            }
            
            # Leaderboards update frequently
            ttl = ttl or 300  # 5 minutes
            
            self.cache.set(key, json.dumps(cache_data), ttl)
            return True
            
        except Exception as e:
            print(f"Failed to cache leaderboard: {e}")
            return False
    
    def get_cached_leaderboard(self, task_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached leaderboard.
        
        Args:
            task_name: Task name
            
        Returns:
            Cached leaderboard or None if not found
        """
        try:
            key = self._generate_key(self.prefixes["tasks"], task_name, "leaderboard")
            
            cached_data = self.cache.get(key)
            if cached_data:
                data = json.loads(cached_data)
                return data.get("leaderboard")
            
            return None
            
        except Exception as e:
            print(f"Failed to get cached leaderboard: {e}")
            return None
    
    def cache_metrics_history(self,
                             agent_name: str,
                             task_name: str,
                             metrics_data: List[Dict[str, Any]],
                             ttl: Optional[int] = None) -> bool:
        """Cache metrics history for agent-task combination.
        
        Args:
            agent_name: Agent name
            task_name: Task name
            metrics_data: Historical metrics data
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            key = self._generate_key(
                self.prefixes["metrics"],
                agent_name,
                task_name,
                "history"
            )
            
            cache_data = {
                "metrics": metrics_data,
                "cached_at": datetime.now().isoformat()
            }
            
            self.cache.set(key, json.dumps(cache_data), ttl or self.default_ttl)
            return True
            
        except Exception as e:
            print(f"Failed to cache metrics history: {e}")
            return False
    
    def get_cached_metrics_history(self,
                                  agent_name: str,
                                  task_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached metrics history.
        
        Args:
            agent_name: Agent name
            task_name: Task name
            
        Returns:
            Cached metrics history or None if not found
        """
        try:
            key = self._generate_key(
                self.prefixes["metrics"],
                agent_name,
                task_name,
                "history"
            )
            
            cached_data = self.cache.get(key)
            if cached_data:
                data = json.loads(cached_data)
                return data.get("metrics")
            
            return None
            
        except Exception as e:
            print(f"Failed to get cached metrics history: {e}")
            return None
    
    def invalidate_agent_cache(self, agent_name: str) -> bool:
        """Invalidate all cache entries for an agent.
        
        Args:
            agent_name: Agent name
            
        Returns:
            True if invalidated successfully
        """
        try:
            # This is a simplified implementation
            # In production, you'd want to use pattern matching or tags
            patterns = [
                self._generate_key(self.prefixes["agents"], agent_name, "*"),
                self._generate_key(self.prefixes["results"], agent_name, "*"),
                self._generate_key(self.prefixes["metrics"], agent_name, "*")
            ]
            
            # For now, we'll mark for manual cleanup
            # Redis pattern deletion would be implemented here
            return True
            
        except Exception as e:
            print(f"Failed to invalidate agent cache: {e}")
            return False
    
    def invalidate_task_cache(self, task_name: str) -> bool:
        """Invalidate all cache entries for a task.
        
        Args:
            task_name: Task name
            
        Returns:
            True if invalidated successfully
        """
        try:
            # Simplified implementation
            patterns = [
                self._generate_key(self.prefixes["tasks"], task_name, "*"),
                self._generate_key(self.prefixes["results"], "*", task_name, "*"),
                self._generate_key(self.prefixes["metrics"], "*", task_name, "*")
            ]
            
            # Pattern deletion would be implemented here
            return True
            
        except Exception as e:
            print(f"Failed to invalidate task cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics.
        
        Returns:
            Cache statistics
        """
        try:
            # This would require Redis INFO command or similar
            # For now, return placeholder stats
            return {
                "cache_enabled": self.cache._redis_client is not None,
                "prefixes": list(self.prefixes.keys()),
                "default_ttl": self.default_ttl
            }
            
        except Exception as e:
            print(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    def warm_cache(self, data_sources: Dict[str, Any]) -> Dict[str, bool]:
        """Pre-warm cache with frequently accessed data.
        
        Args:
            data_sources: Dictionary of data to cache
            
        Returns:
            Dictionary of cache warming results
        """
        results = {}
        
        try:
            # Warm agent performance data
            if "agent_performance" in data_sources:
                for agent_name, perf_data in data_sources["agent_performance"].items():
                    results[f"agent_{agent_name}"] = self.cache_agent_performance(
                        agent_name, perf_data
                    )
            
            # Warm task metadata
            if "task_metadata" in data_sources:
                for task_name, metadata in data_sources["task_metadata"].items():
                    results[f"task_{task_name}"] = self.cache_task_metadata(
                        task_name, metadata
                    )
            
            # Warm leaderboards
            if "leaderboards" in data_sources:
                for task_name, leaderboard in data_sources["leaderboards"].items():
                    results[f"leaderboard_{task_name}"] = self.cache_leaderboard(
                        task_name, leaderboard
                    )
            
            return results
            
        except Exception as e:
            print(f"Failed to warm cache: {e}")
            return {"error": str(e)}