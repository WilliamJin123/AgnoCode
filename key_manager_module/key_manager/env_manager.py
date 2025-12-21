import atexit
import threading
import time
import queue
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from threading import Lock, Event
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict, deque
from enum import Enum
import importlib
import libsql_client # The Turso client

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markup import escape

import logging
try:
    from .log_config import configure_logging
except ImportError:
    from log_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    PER_MODEL = "per_model"  # Cerebras, Groq, Gemini
    GLOBAL = "global"        # OpenRouter (Shared limits across all models)

def get_agno_model_class(provider: str):
    """
    Dynamically maps a provider string to the actual Agno model class.
    """
    p_low = provider.lower()

    overrides = {
        "openai": "OpenAI",
        "google": "Gemini",
        "gemini": "Gemini",
        "azure": "AzureOpenAI",
        "aws": "Bedrock",
        "openrouter": "OpenRouter"
    }

    class_name = overrides.get(p_low, p_low.capitalize())

    module_path = "google" if p_low in ["google", "gemini"] else p_low
    
    try:
        module = importlib.import_module(f"agno.models.{module_path}")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Agno class '{class_name}' not found for provider '{provider}': {e}")

# --- CONFIGURATION DATA ---

@dataclass
class RateLimits:
    """Rate limits for a provider"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None

@dataclass
class UsageSnapshot:
    """Standardized view of usage counters"""
    rpm: int = 0
    rph: int = 0
    rpd: int = 0
    tpm: int = 0
    tph: int = 0
    tpd: int = 0
    total_requests: int = 0
    total_tokens: int = 0

    def __add__(self, other):
        """Allow summing snapshots for aggregation"""
        if not isinstance(other, UsageSnapshot): return NotImplemented
        return UsageSnapshot(
            self.rpm + other.rpm, self.rph + other.rph, self.rpd + other.rpd,
            self.tpm + other.tpm, self.tph + other.tph, self.tpd + other.tpd,
            self.total_requests + other.total_requests, self.total_tokens + other.total_tokens
        )

# --- DATABASE LAYER ---

class UsageDatabase:
    """Handles Online LibSQL (Turso) persistence for API usage"""
    def __init__(self, db_url: Optional[str] = None, auth_token: Optional[str] = None):
        raw_url = db_url or os.getenv("TURSO_DATABASE_URL")
        if raw_url:
            if raw_url.startswith("libsql://"):
                raw_url = raw_url.replace("libsql://", "https://")
            elif raw_url.startswith("wss://"):
                raw_url = raw_url.replace("wss://", "https://")
        
        self.db_url = raw_url
        self.auth_token = auth_token or os.getenv("TURSO_AUTH_TOKEN")
        self._init_db()
    
    def _get_client(self, synchronous = True):
        if synchronous:
            return libsql_client.create_client_sync(url=self.db_url, auth_token=self.auth_token)
        else:
            return libsql_client.create_client(url=self.db_url, auth_token=self.auth_token)

    def _init_db(self):
        with self._get_client() as client:
            # Create table for request logs
            # client.execute("PRAGMA journal_mode=WAL;")
            client.execute("""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    model TEXT,
                    api_key_suffix TEXT,
                    timestamp REAL,
                    tokens INTEGER
                )
            """)
            client.execute("CREATE INDEX IF NOT EXISTS idx_key_usage ON usage_logs(provider, api_key_suffix, timestamp)")
            client.execute("CREATE INDEX IF NOT EXISTS idx_cleanup ON usage_logs(timestamp)")
            client.execute("CREATE INDEX IF NOT EXISTS idx_model_reporting ON usage_logs(provider, model, timestamp)")     

    def load_history(self, provider: str, api_key: str, seconds_lookback: int) -> List[tuple[str, float, int]]:
        """Load history SPECIFIC to this Provider + Model combination"""
        suffix = api_key[-8:] if len(api_key) > 8 else api_key
        cutoff = time.time() - seconds_lookback    
        with self._get_client() as client:
            rs = client.execute(
                """
                SELECT model, timestamp, tokens FROM usage_logs 
                WHERE provider = ? AND api_key_suffix = ? AND timestamp > ?
                ORDER BY timestamp ASC
                """,
                (provider, suffix, cutoff)
            )
            return rs.rows

    def load_provider_history(self, provider: str, seconds_lookback: int):
        """Optimization: Load everything for the provider in ONE call"""
        cutoff = time.time() - seconds_lookback
        with self._get_client() as client:
            rs = client.execute(
                "SELECT api_key_suffix, model, timestamp, tokens FROM usage_logs "
                "WHERE provider = ? AND timestamp > ?",
                (provider, cutoff)
            )
            return rs.rows

    def prune_old_records(self, days_retention: int = 3):
        """Delete records older than retention period to keep DB small (3 days)"""
        cutoff = time.time() - (days_retention * 86400)
        with self._get_client() as client:
            client.execute("DELETE FROM usage_logs WHERE timestamp < ?", (cutoff,))

# --- ASYNC LOGGER ---
class AsyncUsageLogger:
    """Decouples Turso DB writes from the main thread using batching."""
    def __init__(self, db: UsageDatabase):
        self.db = db
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def log(self, provider: str, model: str, api_key: str, tokens: int):
        self.queue.put((provider, model, api_key, time.time(), tokens))

    def _writer_loop(self):
        batch = []
        client = self.db._get_client()
        insert_sql = (
            "INSERT INTO usage_logs (provider, model, api_key_suffix, timestamp, tokens) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        while not self._stop_event.is_set() or not self.queue.empty(): #always empty queue
            try:
                # Wait for the first item (block for up to 1 second)
                record = self.queue.get(timeout=1.0)
                # Parse record for batch formatting
                provider, model, full_key, ts, tokens = record
                suffix = full_key[-8:] if len(full_key) > 8 else full_key
                batch.append((insert_sql, (provider, model, suffix, ts, tokens)))
                
                # Drain queue up to 50 items to batch write
                while len(batch) < 50:
                    try:
                        r = self.queue.get_nowait()
                        p, m, k, t, tok = r
                        s = k[-8:] if len(k) > 8 else k
                        batch.append((insert_sql, (p, m, s, t, tok)))
                    except queue.Empty:
                        break
                
                if batch:
                    client.batch(batch)
                    batch.clear()
                
            except queue.Empty:
                continue
            except Exception as e:
                # print(f"Logging thread error: {e}")
                logger.exception("Logging thread error", exc_info=e)
                time.sleep(2)
        
        if batch:
            try:
                client.batch(batch)
            except:
                pass
        
        client.close()
        
    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning("AsyncUsageLogger thread did not exit cleanly within timeout.")

# --- USAGE TRACKING ---
@dataclass
class UsageBucket:
    """Tracks counters for a SINGLE model context"""
    requests_minute: deque[float] = field(default_factory=deque)
    requests_hour: deque[float] = field(default_factory=deque)
    requests_day: deque[float] = field(default_factory=deque)
    
    tokens_minute: deque[tuple[float, int]] = field(default_factory=deque)
    tokens_hour: deque[tuple[float, int]] = field(default_factory=deque)
    tokens_day: deque[tuple[float, int]] = field(default_factory=deque)
    
    total_requests: int = 0
    total_tokens: int = 0
    
    pending_tokens: int = 0
    
    def clean(self):
        """Clean old entries based on current time"""
        now = time.time()
        cutoffs = (now - 60, now - 3600, now - 86400)
        
        for d, cut in zip(
            [self.requests_minute, self.requests_hour, self.requests_day], cutoffs
        ):
            while d and d[0] <= cut: d.popleft()
            
        for d, cut in zip(
            [self.tokens_minute, self.tokens_hour, self.tokens_day], cutoffs
        ):
            while d and d[0][0] <= cut: d.popleft()
    
    def add(self, tokens: int, timestamp: float):
        self.requests_minute.append(timestamp)
        self.requests_hour.append(timestamp)
        self.requests_day.append(timestamp)
        self.total_requests += 1
        
        if tokens > 0:
            self.tokens_minute.append((timestamp, tokens))
            self.tokens_hour.append((timestamp, tokens))
            self.tokens_day.append((timestamp, tokens))
            self.total_tokens += tokens
    
    def check_limits(self, limits: RateLimits, estimated_tokens: int) -> bool:
        self.clean()
        if len(self.requests_minute) >= limits.requests_per_minute: return False
        if len(self.requests_hour) >= limits.requests_per_hour: return False
        if len(self.requests_day) >= limits.requests_per_day: return False
        
        current_tpm = sum(t[1] for t in self.tokens_minute) + self.pending_tokens
        current_tph = sum(t[1] for t in self.tokens_hour) + self.pending_tokens
        current_tpd = sum(t[1] for t in self.tokens_day) + self.pending_tokens
        
        if limits.tokens_per_minute and (current_tpm + estimated_tokens > limits.tokens_per_minute): return False
        if limits.tokens_per_hour and (current_tph + estimated_tokens > limits.tokens_per_hour): return False
        if limits.tokens_per_day and (current_tpd + estimated_tokens > limits.tokens_per_day): return False
        
        return True
    
    def reserve(self, tokens: int):
        """Lock in estimated tokens"""
        self.pending_tokens += tokens
    
    def commit(self, actual_tokens: int, reserved_tokens: int, timestamp: float):
        """Remove reservation and add actual usage"""
        self.pending_tokens -= reserved_tokens
        if self.pending_tokens < 0: self.pending_tokens = 0 # Safety floor
        self.add(actual_tokens, timestamp)
    
    def get_snapshot(self) -> UsageSnapshot:
        """Return current counts as a clean snapshot"""
        self.clean()
        return UsageSnapshot(
            rpm=len(self.requests_minute),
            rph=len(self.requests_hour),
            rpd=len(self.requests_day),
            tpm=sum(t[1] for t in self.tokens_minute),
            tph=sum(t[1] for t in self.tokens_hour),
            tpd=sum(t[1] for t in self.tokens_day),
            total_requests=self.total_requests,
            total_tokens=self.total_tokens
        )
    

@dataclass
class KeyUsage:
    """Represents an API Key and holds multiple UsageBuckets (one per model)"""
    api_key: str
    strategy: RateLimitStrategy
    buckets: Dict[str, UsageBucket] = field(default_factory=lambda: defaultdict(UsageBucket))
    global_bucket: UsageBucket = field(default_factory=UsageBucket)
    last_429: float = 0.0
    
    def record_usage(self, model_id: str, tokens: int, timestamp: float = None):
        ts = timestamp if timestamp else time.time()
        self.buckets[model_id].add(tokens, ts)
        if self.strategy == RateLimitStrategy.GLOBAL:
            self.global_bucket.add(tokens, ts)
        
    def can_use_model(self, model_id: str, limits: RateLimits, estimated_tokens: int = 1000) -> bool:
        """Check limits based on the provider's strategy"""
        if self.strategy == RateLimitStrategy.GLOBAL:
            return self.global_bucket.check_limits(limits, estimated_tokens)
        else: # Per-Model Limits
            return self.buckets[model_id].check_limits(limits, estimated_tokens)

    def get_total_snapshot(self) -> UsageSnapshot:
        if self.strategy == RateLimitStrategy.GLOBAL:
            return self.global_bucket.get_snapshot()
        total = UsageSnapshot()
        for b in self.buckets.values():
            total = total + b.get_snapshot()
        return total
    
    def reserve(self, model_id: str, tokens: int):
        self.buckets[model_id].reserve(tokens)
        if self.strategy == RateLimitStrategy.GLOBAL:
            self.global_bucket.reserve(tokens)
    
    def commit(self, model_id: str, actual_tokens: int, reserved_tokens: int, timestamp: float = None):
        ts = timestamp if timestamp else time.time()
        self.buckets[model_id].commit(actual_tokens, reserved_tokens, ts)
        if self.strategy == RateLimitStrategy.GLOBAL:
            self.global_bucket.commit(actual_tokens, reserved_tokens, ts)


    def is_cooling_down(self, cooldown_seconds: int = 30) -> bool:
        """Returns True if the key is still in its 30s penalty box."""
        if self.last_429 == 0: return False
        return (time.time() - self.last_429) < cooldown_seconds

    def trigger_cooldown(self):
        """Mark this key as rate-limited."""
        self.last_429 = time.time()

# --- 4. STATS DATA TRANSFER OBJECTS (DTOs) ---

@dataclass
class KeySummary:
    index: int; suffix: str; snapshot: UsageSnapshot
@dataclass
class GlobalStats:
    total: UsageSnapshot; keys: List[KeySummary]
@dataclass
class KeyDetailedStats:
    index: int; suffix: str; total: UsageSnapshot; breakdown: Dict[str, UsageSnapshot]
@dataclass
class ModelAggregatedStats:
    model_id: str; total: UsageSnapshot; keys: List[KeySummary]
    
class RotatingKeyManager:
    """Manages API key rotation with rate limiting"""
    CLEANUP_INTERVAL = 55  # seconds
    
    def __init__(self, api_keys: List[str], provider_name: str, 
                 strategy: RateLimitStrategy, db: UsageDatabase):
        self.provider_name = provider_name
        self.strategy = strategy
        self.keys = [KeyUsage(api_key=k, strategy=strategy) for k in api_keys]
        self.current_index = 0
        self.lock = Lock()
        
        self.db = db
        self.logger = AsyncUsageLogger(self.db)
        self._hydrate()

        self._stop_event = Event()
        self._start_cleanup()
        atexit.register(self.stop)
        # print(f"[{provider_name}] Initialized with {len(self.keys)} keys.")
        logger.info("Initialized %d keys for provider %s.", len(self.keys), provider_name)
    
    def force_rotate_index(self):
        """
        Force the internal pointer to increment. 
        Useful when a key hits a 429 despite local checks passing.
        """
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.keys)
    
    def _hydrate(self):
        # print(f"[{self.provider_name}] Loading history...")
        logger.debug("Loading history for provider %s.", self.provider_name)
        all_history = self.db.load_provider_history(self.provider_name, 86400)
        if not all_history:
            logger.info("No history found in Turso for %s.", self.provider_name)
            return
        
        key_map = {k.api_key[-8:]: k for k in self.keys}
        count = 0
        for row in all_history:
            suffix, model_id, ts, tokens = row
            
            if suffix in key_map:
                key_map[suffix].record_usage(model_id, tokens=tokens, timestamp=ts)
                count += 1
        logger.info("Hydrated %d records across %d keys for %s.", 
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
        self.logger.stop() # Flush logs
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
        self.logger.log(self.provider_name, model_id, key_obj.api_key, actual_tokens)
                
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
        
class RotatingCredentialsMixin:
    """
    Mixin that handles key rotation, 429 detection, and 30s cooldown triggers.
    """
    
    def _rotate_credentials(self) -> KeyUsage:
        key_usage: KeyUsage = self.wrapper.get_key_usage(
            model_id=self.id,
            estimated_tokens=self._estimated_tokens,
            wait=self._rotating_wait,
            timeout=self._rotating_timeout
        )
        self.api_key = key_usage.api_key
        
        if hasattr(self, "client"): self.client = None
        if hasattr(self, "async_client"): self.async_client = None
        if hasattr(self, "gemini_client"): self.gemini_client = None

        return key_usage

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Heuristic to detect rate limits across different providers"""
        err_str = str(e).lower()
        # Common indicators of a rate limit
        if "429" in err_str: return True
        if "too many requests" in err_str: return True
        if "rate limit" in err_str: return True
        if "resource exhausted" in err_str: return True 
        return False

    def _get_retry_limit(self):
        user_limit = min(getattr(self, '_max_retries', 5), len(self.wrapper.manager.keys) - 1)
        return user_limit

    def invoke(self, *args, **kwargs):
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:  
                return super().invoke(*args, **kwargs)
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    # print(f" 429 Hit on key ...{self.api_key[-8:]} (Sync). Rotating and retrying ({attempt+1}/{limit})...")
                    logger.warning("429 Hit on key %s (Sync). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e

    async def ainvoke(self, *args, **kwargs):
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:
                return await super().ainvoke(*args, **kwargs)
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    # print(f" 429 Hit on key ...{self.api_key[-8:]} (Async). Rotating and retrying ({attempt+1}/{limit})...")
                    logger.warning("429 Hit on key %s (Async). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e
    
    def invoke_stream(self, *args, **kwargs):
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:  
                yield from super().invoke_stream(*args, **kwargs)
                return
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    # print(f" 429 Hit on key ...{self.api_key[-8:]} (Sync Stream). Rotating and retrying ({attempt+1}/{limit})...")
                    logger.warning("429 Hit on key %s (Sync Stream). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e

    async def ainvoke_stream(self, *args, **kwargs):
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:
                async for chunk in super().ainvoke_stream(*args, **kwargs): 
                    yield chunk
                return
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    # print(f" 429 Hit on key ...{self.api_key[-8:]} (Async Stream). Rotating and retrying ({attempt+1}/{limit})...")
                    logger.warning("429 Hit on key %s (Async Stream). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e

class MultiProviderWrapper:
    """Wrapper for Agno models with rotating API keys"""
    PROVIDER_STRATEGIES = {
        'cerebras': RateLimitStrategy.PER_MODEL,
        'groq': RateLimitStrategy.PER_MODEL,
        'gemini': RateLimitStrategy.PER_MODEL,
        'openrouter': RateLimitStrategy.GLOBAL,
    }
    
    MODEL_LIMITS = {
        'cerebras': {
            'gpt-oss-120b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
            'llama3.1-8b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
            'llama-3.3-70b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
            'qwen-3-32b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
            'qwen-3-235b-a22b-instruct-2507': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
            'zai-glm-4.6': RateLimits(10, 100, 100, 150000, 1000000, 1000000),
        },
        'groq': {
            'allam-2-7b': RateLimits(30, 1800, 7000, 6000, 360000, 500000),
            'groq/compound': RateLimits(30, 250, 250, 70000, None, None),
            'groq/compound-mini': RateLimits(30, 250, 250, 70000, None, None),
            'llama-3.1-8b-instant': RateLimits(30, 1800, 14400, 6000, 360000, 500000),
            'llama-3.3-70b-versatile': RateLimits(30, 1000, 1000, 12000, 720000, 100000),
            'meta-llama/llama-4-maverick-17b-128e-instruct': RateLimits(30, 1000, 1000, 6000, 360000, 500000),
            'meta-llama/llama-4-scout-17b-16e-instruct': RateLimits(30, 1000, 1000, 30000, 1800000, 500000),
            'meta-llama/llama-guard-4-12b': RateLimits(30, 1800, 14400, 15000, 900000, 500000),
            'meta-llama/llama-prompt-guard-2-22m': RateLimits(30, 1800, 14400, 15000, 900000, 500000),
            'meta-llama/llama-prompt-guard-2-86m': RateLimits(30, 1800, 14400, 15000, 900000, 500000),
            'moonshotai/kimi-k2-instruct': RateLimits(60, 1000, 1000, 10000, 600000, 300000),
            'moonshotai/kimi-k2-instruct-0905': RateLimits(60, 1000, 1000, 10000, 600000, 300000),
            'openai/gpt-oss-120b': RateLimits(30, 1000, 1000, 8000, 480000, 200000),
            'openai/gpt-oss-20b': RateLimits(30, 1000, 1000, 8000, 480000, 200000),
            'openai/gpt-oss-safeguard-20b': RateLimits(30, 1000, 1000, 8000, 480000, 200000),
            'playai-tts': RateLimits(10, 100, 100, 1200, 72000, 3600),
            'playai-tts-arabic': RateLimits(10, 100, 100, 1200, 72000, 3600),
            'qwen/qwen3-32b': RateLimits(60, 1000, 1000, 6000, 360000, 500000),
            'whisper-large-v3': RateLimits(20, 2000, 2000),
            'whisper-large-v3-turbo': RateLimits(20, 2000, 2000),
        },
        'gemini': {
            'gemini-2.5-flash': RateLimits(5, 300, 20, 250000, 15000000),
            'gemini-2.5-flash-lite': RateLimits(10, 600, 20, 250000, 15000000),
            'gemini-2.5-flash-tts': RateLimits(3, 180, 10, 10000, 600000),
            'gemini-robotics-er-1.5-preview': RateLimits(10, 600, 250, 250000, 15000000),
            'gemma-3-12b': RateLimits(30, 1800, 14400, 15000, 900000),
            'gemma-3-1b': RateLimits(30, 1800, 14400, 15000, 900000),
            'gemma-3-27b': RateLimits(30, 1800, 14400, 15000, 900000),
            'gemma-3-2b': RateLimits(30, 1800, 14400, 15000, 900000),
            'gemma-3-4b': RateLimits(30, 1800, 14400, 15000, 900000),
        },
        'openrouter': {
            'default': RateLimits(20, 50, 50),
        },
    
    }
    
    _RotatingClass = None
    
    @staticmethod
    def load_api_keys(provider: str, env_file: Optional[str] = None) -> List[str]:
        """Load API keys from environment variables"""
        if env_file:
            env_path = Path(env_file).resolve()
            if not env_path.exists():
                # print(f"Warning: The provided env_file '{env_path}' does not exist.")
                logger.warning("The provided env_file '%s' does not exist.", env_path)
        else:
            env_path = Path.cwd() / ".env"
        load_dotenv(dotenv_path=env_path, override=True)
        num_keys_var = f"NUM_{provider.upper()}"
        num_keys = os.getenv(num_keys_var)
        if not num_keys:
            raise ValueError(f"Environment variable '{num_keys_var}' not found.")
        try:
            num_keys = int(num_keys)
        except ValueError:
            raise ValueError(f"'{num_keys_var}' must be an integer, got: {num_keys}")

        api_keys = []
        for i in range(1, num_keys + 1):
            key_var = f"{provider.upper()}_API_KEY_{i}"
            key = os.getenv(key_var)
            if not key: raise ValueError(f"Missing API key: {provider.upper()}_API_KEY_{i}")
            api_keys.append(key)
        return api_keys
    
    @classmethod
    def from_env(cls, provider: str, default_model_id: str, 
        env_file: Optional[str] = None,
        db_url: Optional[str] = None,
        db_token: Optional[str] = None,
        debug: bool = False, **kwargs):
        
        model_class = get_agno_model_class(provider)
        api_keys = cls.load_api_keys(provider, env_file)
        db_url = db_url or os.getenv("TURSO_DATABASE_URL")
        db_token = db_token or os.getenv("TURSO_AUTH_TOKEN")
        return cls(provider, model_class, default_model_id, 
                   api_keys, db_url, db_token, debug, **kwargs)
    
    def __init__(self, provider: str, model_class: Any, 
                default_model_id: str, api_keys: List[str], 
                db_url: Optional[str] = None, db_token: Optional[str] = None,
                debug: bool = False, **kwargs):
        self.provider = provider.lower()
        self.model_class = model_class
        self.default_model_id = default_model_id
        self.model_kwargs = kwargs
        self.toggle_debug(debug)
        self.db = UsageDatabase(db_url, db_token)
        self.strategy = self.PROVIDER_STRATEGIES.get(self.provider, RateLimitStrategy.PER_MODEL)
        self.manager = RotatingKeyManager(api_keys, self.provider, self.strategy, self.db)
        self._model_cache = {}
        self.console = Console()

    def toggle_debug(self, enable: bool):
        """
        Dynamically switches logging verbosity for this module.
        enable=True  -> Shows detailed rotation/reservation logs (DEBUG)
        enable=False -> Shows only key info/warnings (INFO)
        """
        level = logging.DEBUG if enable else logging.INFO
        # Set the logger for the current file context
        logger.setLevel(level)
        
        status = "ENABLED" if enable else "DISABLED"
        logger.info(f"Debug logging {status} for {self.provider}")

    def get_key_usage(self, model_id: str = None, estimated_tokens: int = 1000, wait: bool = True, timeout: float = 10):
        """Finds a valid key"""
        mid = model_id or self.default_model_id
        strategy = self.PROVIDER_STRATEGIES.get(self.provider, RateLimitStrategy.PER_MODEL)
        provider_limits = self.MODEL_LIMITS.get(self.provider, {})
        limits = provider_limits.get(mid, provider_limits.get('default', RateLimits(10, 100, 1000)))

        start = time.time()
        while True:
            key_usage = self.manager.get_key(mid, limits, estimated_tokens)
            if key_usage: 
                return key_usage
            if not wait:
                    raise RuntimeError(f"No available API keys for {self.provider}/{mid} (wait=False)") 
            if time.time() - start > timeout:
                raise RuntimeError(f"Timeout: No available API keys for {self.provider}/{mid} after {timeout}s")
            time.sleep(0.5)

    def get_model(self, estimated_tokens: int = 1000, wait: bool = True, timeout: float = 10, max_retries: int = 5, **kwargs):
        """Dynamically creates a rotating model for ANY provider."""
        if self._RotatingClass is None:
            self._RotatingClass = type(
                f"Rotating{self.model_class.__name__}",
                (RotatingCredentialsMixin, self.model_class),
                {}
            )
        RotatingProviderClass = self._RotatingClass
        # 2. Get Initial Key
        model_id = kwargs.get('id', self.default_model_id)  
        initial_key_usage = self.get_key_usage(model_id, estimated_tokens, wait=wait, timeout=timeout)
        final_kwargs = {**self.model_kwargs, **kwargs}
        if 'id' not in final_kwargs:
            final_kwargs['id'] = model_id

        model_instance = RotatingProviderClass(
            api_key=initial_key_usage.api_key,
            **final_kwargs
        )

        model_instance.wrapper = self
        model_instance._rotating_wait = wait
        model_instance._rotating_timeout = timeout
        model_instance._estimated_tokens = estimated_tokens
        model_instance._max_retries = max_retries
        
        orig_metrics = getattr(model_instance, "_get_metrics", None)
        def metrics_hook(*args, **kwargs):
            if orig_metrics:
                m = orig_metrics(*args, **kwargs)
            else:
                m = None
            actual = 0
            if m and hasattr(m, 'total_tokens') and m.total_tokens is not None:
                actual = m.total_tokens
            # Retrieve the estimate we set on the instance earlier
            estimate = getattr(model_instance, "_estimated_tokens", 1000)
            self.manager.record_usage(
                key_obj=initial_key_usage,
                model_id=model_id, 
                actual_tokens=actual, 
                estimated_tokens=estimate
            )
            return m
        
        model_instance._get_metrics = metrics_hook
        return model_instance
    
    # --- PRINTING HELPERS ---
    
    def _create_usage_table(self, title: str, data: List[tuple[str, UsageSnapshot]]) -> Table:
        """
        Generates a standardized table for usage stats.
        data format: [(Label, Snapshot), ...]
        """
        # Palette
        c_title = "#bae1ff"  # Pastel Rose
        c_head  = "#f2f2f2"  # Pastel Cream
        c_req   = "#faa0a0"  # Pastel Periwinkle
        c_tok   = "#e5baff"  # Pastel Peach
        c_border= "#B9B9B9"  # Muted Grey
        c_identifier = "#7cd292"  # Soft Mauve
        
        table = Table(
            title=title, 
            box=box.ROUNDED, 
            expand=False, 
            title_style=f"bold {c_title}",
            title_justify="left",
            border_style=c_border,
            header_style=f"{c_head}"
        )

        # Define Columns
        table.add_column("Identifier", style=f"bold {c_identifier}", no_wrap=True)
        table.add_column("Requests (m/h/d)",  justify="center", style=c_req, no_wrap=True)
        table.add_column("Tokens (m/h/d)",  justify="center", style=c_tok, no_wrap=True)
        table.add_column("Total Reqs", justify="center", style=f"{c_req}", no_wrap=True)
        table.add_column("Total Tokens", justify="center", style=f"bold {c_tok}", no_wrap=True)

        for label, s in data:
            req_str = f"{s.rpm} / {s.rph} / {s.rpd}"
            tok_str = f"{s.tpm:,} / {s.tph:,} / {s.tpd:,}"
            
            table.add_row(
                label,
                req_str,
                tok_str,
                f"{s.total_requests}",
                f"{s.total_tokens:,}"
            )
        return table

    def print_global_stats(self):
        stats = self.manager.get_global_stats()
        
        # 1. Prepare Data for the Table
        rows = []
        for k in stats.keys:
            label = f"Key #{k.index+1} (..{k.suffix})"
            rows.append((label, k.snapshot))
            
        # 2. Create and Print Table
        table = self._create_usage_table(
            title=f"GLOBAL STATS: {escape(self.provider.upper())}", 
            data=rows
        )
        
        # 3. Add a Summary Footer (using a Panel for the Total)
        total_s = stats.total
        grid = Table.grid(padding=(0, 4)) 
        grid.add_column(style="#e0e0e0") # Label Color
        grid.add_column(style="bold", justify="left") # Value Color

        grid.add_row("Total Requests:", f"[{'#faa0a0'}]{total_s.total_requests}[/]")
        grid.add_row("Total Tokens:",   f"[{'#e5baff'}]{total_s.total_tokens:,}[/]")
        
        self.console.print()
        self.console.print(Panel(
            grid, 
            title="[bold #bae1ff] AGGREGATE TOTALS [/]", 
            border_style="#bae1ff",
            expand=False
        ))
        self.console.print(table)

    def print_key_stats(self, identifier: Union[int, str]):
        stats = self.manager.get_key_stats(identifier)
        if not stats:
            self.console.print(f"[bold red]Key not found:[/][white] {identifier}[/]")
            return
        
        self.console.print()
        self.console.rule(f"[bold]Key Report: {stats.suffix}[/]")
        
        # 1. Total Snapshot Panel
        s = stats.total
        grid = Table.grid(padding=(0, 4))
        grid.add_column(style="#e0e0e0")
        grid.add_column(justify="left")

        grid.add_row("Total Requests:", f"[{'#faa0a0'}]{s.total_requests}[/]")
        grid.add_row("Total Tokens:",   f"[{'#e5baff'}]{s.total_tokens:,}[/]")
        
        self.console.print(Panel(
            grid, 
            title=f"[bold #97e3e9]Key #{stats.index+1} Overview[/]", 
            border_style="#bae1ff",
            expand=False
        ))

        # 2. Breakdown Table
        if not stats.breakdown:
            self.console.print("[italic dim]No usage recorded for this key yet.[/]")
        else:
            rows = [(model_id, snap) for model_id, snap in stats.breakdown.items()]
            table = self._create_usage_table(title="Breakdown by Model", data=rows)
            self.console.print(table)

    def print_model_stats(self, model_id: str):
        data = self.manager.get_model_stats(model_id)
        
        self.console.print()
        self.console.rule(f"[bold]Model Report: {model_id}[/]", style="#B9B9B9")
        
        # 1. Total Summary
        s = data.total
        self.console.print(f"Total Tokens Consumed: [bold green]{s.total_tokens:,}[/]")
        
        # 2. Contributing Keys Table
        if not data.keys:
            self.console.print("[italic dim]No keys have used this model.[/]")
        else:
            rows = []
            for k in data.keys:
                label = f"Key #{k.index+1} (..{k.suffix})"
                rows.append((label, k.snapshot))
            
            table = self._create_usage_table(title="Contributing Keys", data=rows)
            self.console.print(table)

    def print_granular_stats(self, identifier: Union[int, str], model_id: str):
        data = self.manager.get_granular_stats(identifier, model_id)
        
        if not data:
            self.console.print(f"[bold red]Key '{identifier}' not found.[/]")
            return

        self.console.print()
        if data.snapshot:
            # Re-use the table builder for a single row just for consistency
            label = f"Key #{data.index+1} (..{data.suffix})"
            table = self._create_usage_table(
                title=f"Granular: {model_id}", 
                data=[(label, data.snapshot)]
            )
            self.console.print(table)
        else:
            self.console.print(Panel(
                f"No usage for model [bold]{model_id}[/] on key [bold]..{data.suffix}[/]",
                style="#e5baff",
                border_style="#B9B9B9"
            ))