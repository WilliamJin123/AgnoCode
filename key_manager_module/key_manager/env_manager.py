import atexit
import threading
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from threading import Lock, Event, Timer
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict, deque

# --- 1. CONFIGURATION DATA ---

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

# --- 2. DATABASE LAYER ---

class UsageDatabase:
    """Handles SQLite persistence for API usage"""
    def __init__(self, db_path: str = "api_usage.db"):
        self.db_path = db_path
        self._init_db()
    def _init_db(self):
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            # Create table for request logs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    model TEXT,
                    api_key_suffix TEXT,
                    timestamp REAL,
                    tokens INTEGER
                )
            """)
            # 1. Operational: Loading history for a specific key
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_key_usage 
                ON usage_logs(provider, api_key_suffix, timestamp)
            """)

            # 2. Maintenance: Pruning old records
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cleanup 
                ON usage_logs(timestamp)
            """)

            # 3. Analytics: Reporting usage by model (ignoring key)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_reporting 
                ON usage_logs(provider, model, timestamp)
            """)
            

    def record_usage(self, provider: str, model: str, api_key: str, tokens: int):
        """Log a single request to the DB"""
        suffix = api_key[-8:] if len(api_key) > 8 else api_key
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(
                "INSERT INTO usage_logs (provider, model, api_key_suffix, timestamp, tokens) VALUES (?, ?, ?, ?, ?)",
                (provider, model, suffix, time.time(), tokens)
            )

    def load_history(self, provider: str, api_key: str, seconds_lookback: int) -> List[tuple[float, int]]:
        """Load history SPECIFIC to this Provider + Model combination"""
        suffix = api_key[-8:] if len(api_key) > 8 else api_key
        cutoff = time.time() - seconds_lookback
        
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cursor = conn.execute(
                """
                SELECT model, timestamp, tokens FROM usage_logs 
                WHERE provider = ? AND api_key_suffix = ? AND timestamp > ?
                ORDER BY timestamp ASC
                """,
                (provider, suffix, cutoff)
            )
            return cursor.fetchall()

    def prune_old_records(self, days_retention: int = 3):
        """Delete records older than retention period to keep DB small (3 days)"""
        cutoff = time.time() - (days_retention * 86400)
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute("DELETE FROM usage_logs WHERE timestamp < ?", (cutoff,))

# --- 3. USAGE TRACKING LOGIC ---
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
    
    def clean(self):
        """Clean old entries based on current time"""
        now = time.time()
        self._clean_deque(self.requests_minute, now - 60)
        self._clean_deque(self.requests_hour, now - 3600)
        self._clean_deque(self.requests_day, now - 86400)
        
        self._clean_deque(self.tokens_minute, now - 60)
        self._clean_deque(self.tokens_hour, now - 3600)
        self._clean_deque(self.tokens_day, now - 86400)
    
    def _clean_deque(self, d: deque, cutoff: float):
        while d:
            ts = d[0] if isinstance(d[0], float) else d[0][0]
            if ts > cutoff:
                break
            d.popleft()
    
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
    
    def check_limits(self, limits: RateLimits, estimated_tokens: int) -> bool:
        self.clean()
        
        if len(self.requests_minute) >= limits.requests_per_minute: return False
        if len(self.requests_hour) >= limits.requests_per_hour: return False
        if len(self.requests_day) >= limits.requests_per_day: return False
        
        current_tpm = sum(t[1] for t in self.tokens_minute)
        if limits.tokens_per_minute and current_tpm + estimated_tokens > limits.tokens_per_minute: return False
        
        current_tph = sum(t[1] for t in self.tokens_hour)
        if limits.tokens_per_hour and current_tph + estimated_tokens > limits.tokens_per_hour: return False
        
        current_tpd = sum(t[1] for t in self.tokens_day)
        if limits.tokens_per_day and current_tpd + estimated_tokens > limits.tokens_per_day: return False
        
        return True

@dataclass
class KeyUsage:
    """Represents an API Key and holds multiple UsageBuckets (one per model)"""
    api_key: str
    buckets: Dict[str, UsageBucket] = field(default_factory=lambda: defaultdict(UsageBucket))
    
    def record_usage(self, model_id: str, tokens: int, timestamp: float = None):
        ts = timestamp if timestamp else time.time()
        self.buckets[model_id].add(tokens, ts)
    
    def can_use_model(self, model_id: str, limits: RateLimits, estimated_tokens: int = 0) -> bool:
        return self.buckets[model_id].check_limits(limits, estimated_tokens)

    def get_total_snapshot(self) -> UsageSnapshot:
        """Aggregates ALL buckets for this key"""
        total = UsageSnapshot()
        for bucket in self.buckets.values():
            total = total + bucket.get_snapshot()
        return total

# --- 4. STATS DATA TRANSFER OBJECTS (DTOs) ---

@dataclass
class KeySummary:
    index: int
    suffix: str
    snapshot: UsageSnapshot

@dataclass
class GlobalStats:
    total: UsageSnapshot
    keys: List[KeySummary]

@dataclass
class KeyDetailedStats:
    index: int
    suffix: str
    total: UsageSnapshot
    breakdown: Dict[str, UsageSnapshot]

@dataclass
class ModelAggregatedStats:
    model_id: str
    total: UsageSnapshot
    keys: List[KeySummary]

class RotatingKeyManager:
    """Manages API key rotation with rate limiting"""
    CLEANUP_INTERVAL = 60
    
    def __init__(self, api_keys: List[str], provider_name: str, db_path: str = "api_usage.db"):
        self.provider_name = provider_name
        self.keys = [KeyUsage(api_key=k) for k in api_keys]
        self.current_index = 0
        self.lock = Lock()
        
        self.db = UsageDatabase(db_path)
        self._hydrate()

        self._stop_event = Event()
        self._start_cleanup()
        atexit.register(self.stop)
        print(f"[{provider_name}] Initialized with {len(self.keys)} keys.")
    
    def _hydrate(self):
        print(f"[{self.provider_name}] Loading history...")
        for key in self.keys:
            history = self.db.load_history(self.provider_name, key.api_key, 86400) # 24 hours
            for model_id, timestamp, tokens in history:
                key.record_usage(model_id, tokens=tokens, timestamp=timestamp)
    
    def _cleanup_loop(self):
        """Periodically clean deques to prevent memory bloat"""
        while not self._stop_event.is_set():
            with self.lock:
                for key in self.keys:
                    for bucket in key.buckets.values():
                        bucket.clean()
                # try:
                #     self.db.prune_old_records()
                # except: pass
            time.sleep(self.CLEANUP_INTERVAL)
    
    def _start_cleanup(self):
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._stop_event.set()
    
    def get_key(self, model_id: str, limits: RateLimits, estimated_tokens: int = 0) -> Optional[KeyUsage]:
        """Get an available API key that can handle the request"""
        with self.lock:
            for offset in range(len(self.keys)):
                idx = (self.current_index + offset) % len(self.keys)
                key : KeyUsage = self.keys[idx]
                
                if key.can_use_model(model_id, limits, estimated_tokens):
                    self.current_index = idx
                    return key
            return None
    
    def record_usage(self, api_key: Union[str, KeyUsage], model_id: str, tokens_used: int = 0):
        """Record usage for a specific API key"""
        with self.lock:
            key_identifier = api_key if isinstance(api_key, str) else api_key.api_key
            for key in self.keys:
                if key.api_key == key_identifier:
                    key.record_usage(model_id, tokens_used) #timestamp automatically generated as now()
                    self.db.record_usage(self.provider_name, model_id, key.api_key, tokens_used)
                    break  
                
    # --- STATS HELPERS ---
    
    def _find_key(self, identifier: Union[int, str]) -> tuple[Optional[KeyUsage], int]:
        """Locate key by index (int) or suffix/full-key (str)"""
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.keys):
                return self.keys[identifier], identifier
        elif isinstance(identifier, str):
            for i, k in enumerate(self.keys):
                if k.api_key == identifier or k.api_key.endswith(identifier):
                    return k, i
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
            snap = UsageSnapshot() # Default empty
            
            if model_id in key.buckets:
                snap = key.buckets[model_id].get_snapshot()
            
            return KeySummary(index=idx, suffix=suffix, snapshot=snap)

class RotatingCredentialsMixin:
    """
    A universal Mixin that forces a key rotation and client rebuild
    before every single model invocation.
    """
    def _rotate_credentials(self):
        wait = getattr(self, '_rotating_wait', True)
        timeout = getattr(self, '_rotating_timeout', 10)
        estimated_tokens = getattr(self, '_estimated_tokens', 0)

        key_usage: KeyUsage = self.wrapper.get_key_usage(
            model_id=self.id, 
            estimated_tokens=estimated_tokens,
            wait=wait,
            timeout=timeout
        )

        self.api_key = key_usage.api_key
        if hasattr(self, "client"):
            self.client = None
        if hasattr(self, "async_client"):
            self.async_client = None
        if hasattr(self, "gemini_client"):
            self.gemini_client = None

    def invoke(self, *args, **kwargs):
        self._rotate_credentials()
        return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        self._rotate_credentials()
        return await super().ainvoke(*args, **kwargs)
    
    def invoke_stream(self, *args, **kwargs):
        self._rotate_credentials()
        yield from super().invoke_stream(*args, **kwargs)

    async def ainvoke_stream(self, *args, **kwargs):
        self._rotate_credentials()
        async for chunk in super().ainvoke_stream(*args, **kwargs):
            yield chunk

class MultiProviderWrapper:
    """Wrapper for Agno models with rotating API keys"""
    
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
    }
    
    @staticmethod
    def load_api_keys(provider: str, env_file: Optional[str] = None) -> List[str]:
        """Load API keys from environment variables"""
        if env_file:
            env_path = Path(env_file).resolve()
            if not env_path.exists():
                print(f"Warning: The provided env_file '{env_path}' does not exist.")
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
    def from_env(cls, provider: str, model_class: Any, default_model_id: str, 
                 env_file: Optional[str] = None, db_path: str = "api_usage.db", **kwargs):
        api_keys = cls.load_api_keys(provider, env_file)
        return cls(provider, model_class, default_model_id, api_keys, db_path, **kwargs)
    
    def __init__(self, provider: str, model_class: Any, default_model_id: str, api_keys: List[str], 
                db_path: str = "api_usage.db", **kwargs):
        self.provider = provider.lower()
        self.model_class = model_class
        self.default_model_id = default_model_id
        self.model_kwargs = kwargs
        self.manager = RotatingKeyManager(api_keys, self.provider, db_path)
        self._model_cache = {}

    def get_key_usage(self, model_id: str = None, estimated_tokens: int = 0, wait: bool = True, timeout: float = 10):
        """Finds a valid key"""
        mid = model_id or self.default_model_id

        provider_limits = self.MODEL_LIMITS.get(self.provider, {})
        limits = provider_limits.get(mid)
        if not limits:
            print(f"Warning: No limits for {mid}, using default.")
            limits = RateLimits(10, 100, 1000)

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

    def get_model(self, estimated_tokens: int = 0, wait: bool = True, timeout: float = 10, **kwargs):
        """Dynamically creates a rotating model for ANY provider."""
        model_id = kwargs.get('id', self.default_model_id)  
        
        RotatingProviderClass = type(
            f"Rotating{self.model_class.__name__}", 
            (RotatingCredentialsMixin, self.model_class), 
            {}
        )
        # 2. Get Initial Key
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

        orig_metrics = getattr(model_instance, "_get_metrics", None)
        def metrics_hook(*args, **kwargs):
            m = orig_metrics(*args, **kwargs)
            if m and hasattr(m, 'total_tokens'):
                current_key = model_instance.api_key
                self.manager.record_usage(current_key, model_id, m.total_tokens)
            return m
        model_instance._get_metrics = metrics_hook

        return model_instance
    
    # --- PRINTING HELPERS ---
    
    def _print_key_header(self, index: int, suffix: str, extra: str = ""):
        header = f"Key #{index + 1} (...{suffix})"
        if extra:
            header += f" | {extra}"
        print(header)
    
    def _print_snapshot(self, s: UsageSnapshot, indent: str = ""):
        print(f"{indent}Reqs: {s.rpm} m | {s.rph} h | {s.rpd} d  (Total: {s.total_requests})")
        print(f"{indent}Toks: {s.tpm:,} m | {s.tph:,} h | {s.tpd:,} d  (Total: {s.total_tokens:,})")
        
    def print_global_stats(self):
        stats = self.manager.get_global_stats()
        print(f"\n=== GLOBAL STATS ({self.provider.upper()}) ===")
        print(f"Total Keys: {len(stats.keys)}")
        
        print("Aggregate Totals:")
        self._print_snapshot(stats.total)
        print("-" * 40)
        for k in stats.keys:
            print() # Spacer
            self._print_key_header(k.index, k.suffix)
            self._print_snapshot(k.snapshot, indent="  ")
    
    def print_key_stats(self, identifier: Union[int, str]):
        stats = self.manager.get_key_stats(identifier)
        if not stats: return print(f"Key not found: {identifier}")
        
        self._print_key_header(stats.index, stats.suffix, extra="FULL REPORT")
        print()
        
        print("Total Usage (All Models):")
        self._print_snapshot(stats.total)
        
        print("\nModel Breakdown:")
        if not stats.breakdown:
            print("  (No usage recorded)")
        for mid, snap in stats.breakdown.items():
            print(f"\n  [Model: {mid}]")
            self._print_snapshot(snap, indent="    ")
    
    def print_model_stats(self, model_id: str):
        data = self.manager.get_model_stats(model_id)
        print(f"\n=== MODEL STATS ({data.model_id}) ===")
        print("Total Usage (All Keys):")
        self._print_snapshot(data.total)
        
        print(f"\n=== Contributing Keys ===")
        if not data.keys:
            print("  (No keys have used this model)")
            
        for k in data.keys:
            print()
            self._print_key_header(k.index, k.suffix)
            self._print_snapshot(k.snapshot, indent="  ")

    def print_granular_stats(self, identifier: Union[int, str], model_id: str):
        data = self.manager.get_granular_stats(identifier, model_id)
        
        print(f"\n=== PRINTING GRANULAR STATS ===")
        
        if not data:
            print(f"Key identifier '{identifier}' not found.")
            return

        self._print_key_header(data.index, data.suffix, extra=f"Model: {model_id}")
        
        if data.snapshot:
            self._print_snapshot(data.snapshot)
        else:
            print("  (No usage record for this specific combination)")
        