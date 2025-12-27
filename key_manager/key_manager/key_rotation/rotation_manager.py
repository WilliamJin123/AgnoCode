import time
import atexit
import threading
from threading import Lock, Event
from typing import Any, Dict, List, Optional, Union

from ..config.dataclasses import (
    RateLimits, UsageSnapshot,
    GlobalStats, KeySummary,
    KeyDetailedStats, ModelAggregatedStats,
    KeyUsage
)
from ..config.enums import RateLimitStrategy
from ..config.log_config import default_logger
from ..usage.usage_logger import AsyncUsageLogger
from ..usage.db_logic import UsageDatabase

class RotatingKeyManager:
    """Manages API key rotation with rate limiting"""
    CLEANUP_INTERVAL = 55  # seconds
    
    def __init__(self, api_keys: List[str], provider_name: str, 
                 strategy: RateLimitStrategy, db: UsageDatabase, logger = None):
        self.provider_name = provider_name
        self.strategy = strategy
        self.keys = [KeyUsage(api_key=k, strategy=strategy) for k in api_keys]
        self.current_index = 0
        self.lock = Lock()
        
        self.db = db
        self.usage_logger = AsyncUsageLogger(self.db)
        self._hydrate()

        self._stop_event = Event()
        self._start_cleanup()
        atexit.register(self.stop)

        self.logger = logger or default_logger
        self.logger.info("Initialized %d keys for provider %s.", len(self.keys), provider_name)
    
    def force_rotate_index(self):
        """
        Force the internal pointer to increment. 
        Useful when a key hits a 429 despite local checks passing.
        """
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.keys)
    
    def _hydrate(self):
        # print(f"[{self.provider_name}] Loading history...")
        self.logger.debug("Loading history for provider %s.", self.provider_name)
        all_history = self.db.load_provider_history(self.provider_name, 86400)
        if not all_history:
            self.logger.info("No history found in Turso for %s.", self.provider_name)
            return
        
        key_map = {k.api_key[-8:]: k for k in self.keys}
        count = 0
        for row in all_history:
            suffix, model_id, ts, tokens = row
            
            if suffix in key_map:
                key_map[suffix].record_usage(model_id, tokens=tokens, timestamp=ts)
                count += 1
        self.logger.info("Hydrated %d records across %d keys for %s.", 
                    count, len(self.keys), self.provider_name)
    
    def _cleanup_loop(self):
        """Periodically clean deques to prevent memory bloat"""
        while not self._stop_event.is_set():
            with self.lock:
                for key in self.keys:
                    for bucket in key.buckets.values():
                        bucket.clean()
                    if self.strategy == RateLimitStrategy.GLOBAL:
                        key.global_bucket.clean()
                # try:
                #     self.db.prune_old_records()
                # except: pass
            time.sleep(self.CLEANUP_INTERVAL)
    
    def _start_cleanup(self):
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
        self.usage_logger.stop() # Flush logs
        if self._thread.is_alive():
            self._thread.join(timeout=10)
    
    def get_key(self, model_id: str, limits: RateLimits, estimated_tokens: int = 1000) -> Optional[KeyUsage]:
        """Get an available API key that can handle the request"""
        with self.lock:
            for offset in range(len(self.keys)):
                idx = (self.current_index + offset) % len(self.keys)
                key : KeyUsage = self.keys[idx]

                if key.is_cooling_down(30):
                    continue

                if key.can_use_model(model_id, limits, estimated_tokens):
                    key.reserve(model_id, estimated_tokens)
                    self.current_index = idx
                    return key
            return None
    
    def record_usage(self, key_obj: KeyUsage, model_id: str,actual_tokens: int, estimated_tokens: int = 1000):
        """Record usage for a specific API key"""
        with self.lock:
            key_obj.commit(model_id, actual_tokens, estimated_tokens)
        self.usage_logger.log(self.provider_name, model_id, key_obj.api_key, actual_tokens)
                
    # --- STATS HELPERS ---
    
    def _find_key(self, identifier: Union[int, str]) -> tuple[Optional[KeyUsage], int]:
        """Locate key by index (int) or suffix/full-key (str)"""
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.keys):
                return self.keys[identifier], identifier
        elif isinstance(identifier, str):
            for i, k in enumerate(self.keys):
                if k.api_key == identifier or k.api_key.endswith(identifier): return k, i
        return None, -1
    
    def get_global_stats(self) -> GlobalStats:
        """Aggregates usage across all keys and models"""
        total = UsageSnapshot()
        keys_summary = []
        with self.lock:
            for i, key in enumerate(self.keys):
                snap = key.get_total_snapshot()
                total = total + snap
                suffix = key.api_key[-8:] if len(key.api_key)>8 else key.api_key
                keys_summary.append(KeySummary(index=i, suffix=suffix, snapshot=snap))
        return GlobalStats(total=total, keys=keys_summary)
    
    def get_key_stats(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Stats for a specific key, including per-model breakdown"""
        with self.lock:
            key, idx = self._find_key(identifier)
            if not key: return None
            total_snap = key.get_total_snapshot()
            breakdown = {}
            for model, bucket in key.buckets.items():
                breakdown[model] = bucket.get_snapshot()
            suffix = key.api_key[-8:] if len(key.api_key)>8 else key.api_key
            return KeyDetailedStats(index=idx, suffix=suffix, total=total_snap, breakdown=breakdown)
    
    def get_model_stats(self, model_id: str) -> UsageSnapshot:
        """Aggregates stats for ONE model across ALL keys"""
        total = UsageSnapshot()
        contributing_keys = []
        with self.lock:
            for i, key in enumerate(self.keys):
                if model_id in key.buckets:
                    snap = key.buckets[model_id].get_snapshot()
                    total = total + snap
                    suffix = key.api_key[-8:] if len(key.api_key) > 8 else key.api_key
                    contributing_keys.append(KeySummary(index=i, suffix=suffix, snapshot=snap))
        return ModelAggregatedStats(model_id=model_id, total=total, keys=contributing_keys)
    
    def get_granular_stats(self, identifier: Union[int, str], model_id: str) -> Optional[UsageSnapshot]:
        """Specific Key + Specific Model"""
        with self.lock:
            key, idx = self._find_key(identifier)
            if not key: return None
            suffix = key.api_key[-8:] if len(key.api_key) > 8 else key.api_key
            snap = key.buckets[model_id].get_snapshot() if model_id in key.buckets else UsageSnapshot()
            return KeySummary(index=idx, suffix=suffix, snapshot=snap)
 