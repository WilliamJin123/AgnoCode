import atexit
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from threading import Lock, Event, Timer
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv
from collections import deque

@dataclass
class RateLimits:
    """Rate limits for a provider"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None

class UsageDatabase:
    """Handles SQLite persistence for API usage"""
    def __init__(self, db_path: str = "api_usage.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Create table for request logs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    api_key_suffix TEXT,
                    timestamp REAL,
                    tokens INTEGER
                )
            """)
            # Index for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON usage_logs(api_key_suffix)")

    def record_usage(self, provider: str, api_key: str, tokens: int):
        """Log a single request to the DB"""
        suffix = api_key[-8:] if len(api_key) > 8 else api_key
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO usage_logs (provider, api_key_suffix, timestamp, tokens) VALUES (?, ?, ?, ?)",
                (provider, suffix, time.time(), tokens)
            )

    def load_history(self, provider: str, api_key: str, seconds_lookback: int) -> List[tuple[float, int]]:
        """Load recent history for a specific key to populate in-memory deques"""
        suffix = api_key[-8:] if len(api_key) > 8 else api_key
        cutoff = time.time() - seconds_lookback
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, tokens FROM usage_logs 
                WHERE provider = ? AND api_key_suffix = ? AND timestamp > ?
                ORDER BY timestamp ASC
                """,
                (provider, suffix, cutoff)
            )
            return cursor.fetchall()

    def prune_old_records(self, days_retention: int = 3):
        """Delete records older than retention period to keep DB small (3 days)"""
        cutoff = time.time() - (days_retention * 86400)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM usage_logs WHERE timestamp < ?", (cutoff,))

@dataclass
class KeyUsage:
    """Track usage for a single API key"""
    api_key: str
    
    requests_minute: deque[float] = field(default_factory=deque)
    requests_hour: deque[float] = field(default_factory=deque)
    requests_day: deque[float] = field(default_factory=deque)
    
    tokens_minute: deque[tuple[float, int]] = field(default_factory=deque)
    tokens_hour: deque[tuple[float, int]] = field(default_factory=deque)
    tokens_day: deque[tuple[float, int]] = field(default_factory=deque)
    
    total_requests: int = 0
    total_tokens: int = 0
    
    def _clean_old_entries(self, entries: deque, seconds: int):
        """Remove entries older than specified seconds"""
        cutoff = time.time() - seconds
        # Check if the deque stores floats (timestamps) or tuples (timestamp, token_count)
        is_float_deque = not entries or isinstance(entries[0], float)
        
        # Use popleft() to remove expired items from the start (head)
        while entries:
            entry_time = entries[0] if is_float_deque else entries[0][0]
            
            if entry_time > cutoff:
                break
            entries.popleft()
            
    
    def get_current_usage(self) -> Dict[str, int]:
        """Get current usage counts"""
        
        self._clean_old_entries(self.requests_minute, 60)
        self._clean_old_entries(self.requests_hour, 3600)
        self._clean_old_entries(self.requests_day, 86400)
        
        self._clean_old_entries(self.tokens_minute, 60)
        self._clean_old_entries(self.tokens_hour, 3600)
        self._clean_old_entries(self.tokens_day, 86400)
        
        return {
            'requests_per_minute': len(self.requests_minute),
            'requests_per_hour': len(self.requests_hour),
            'requests_per_day': len(self.requests_day),
            'tokens_per_minute': sum(t[1] for t in self.tokens_minute),
            'tokens_per_hour': sum(t[1] for t in self.tokens_hour),
            'tokens_per_day': sum(t[1] for t in self.tokens_day),
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
        }
        
    def can_make_request(self, limits: RateLimits, estimated_tokens: int = 0) -> bool:
        """Check if we can make a request without exceeding limits"""
        usage = self.get_current_usage()
        
        if usage['requests_per_minute'] >= limits.requests_per_minute:
            return False
        if usage['requests_per_hour'] >= limits.requests_per_hour:
            return False
        if usage['requests_per_day'] >= limits.requests_per_day:
            return False
        
        if limits.tokens_per_minute and usage['tokens_per_minute'] + estimated_tokens > limits.tokens_per_minute:
            return False
        if limits.tokens_per_hour and usage['tokens_per_hour'] + estimated_tokens > limits.tokens_per_hour:
            return False
        if limits.tokens_per_day and usage['tokens_per_day'] + estimated_tokens > limits.tokens_per_day:
            return False
        
        return True

    def record_request_in_memory(self, tokens_used: int = 0, timestamp: float = None):
        """Record a request and token usage"""
        now = timestamp if timestamp else time.time()
        
        # Appends to the right end (tail)
        self.requests_minute.append(now)
        self.requests_hour.append(now)
        self.requests_day.append(now)
        self.total_requests += 1
        
        if tokens_used > 0:
            self.tokens_minute.append((now, tokens_used))
            self.tokens_hour.append((now, tokens_used))
            self.tokens_day.append((now, tokens_used))
            self.total_tokens += tokens_used

@dataclass
class UtilizationStats:
    """Formatted strings showing usage vs limits (e.g. '50/1000')"""
    rpm: str
    rph: str
    rpd: str
    
@dataclass
class KeyStats:
    """Statistics for a single API key"""
    key_index: int
    key_suffix: str
    
    # Raw counts
    current_rpm: int
    current_rph: int
    current_rpd: int
    current_tpm: int
    current_tph: int
    current_tpd: int
    
    total_requests: int
    total_tokens: int
    
    # Formatted utilization strings
    utilization: UtilizationStats

@dataclass
class ProviderStats:
    """Overall statistics for the provider wrapper"""
    provider: str
    total_keys: int
    current_key_index: int
    total_requests: int
    total_tokens: int
    keys: List[KeyStats]

class RotatingKeyManager:
    """Manages API key rotation with rate limiting"""
    CLEANUP_INTERVAL_SECONDS = 60
    
    def __init__(self, api_keys: List[str], rate_limits: RateLimits, provider_name: str = "provider", db_path: str = "api_usage.db"):
        self.provider_name = provider_name
        self.rate_limits = rate_limits
        self.keys = [KeyUsage(api_key=key) for key in api_keys]
        self.current_key_index = 0
        self.lock = Lock()

        self.db = UsageDatabase(db_path)
        self._hydrate_keys_from_db()

        self._stop_event = Event()
        self._cleanup_thread = None
        self._start_cleanup_thread()
        atexit.register(self._stop_cleanup_thread)
        
        print(f"Initialized {provider_name} with {len(self.keys)} API keys and background cleanup.")
    
    
    def _hydrate_keys_from_db(self):
        """Load the last 24 hours of usage for each key from SQLite"""
        print(f"Loading history for {self.provider_name}...")
        for key_usage in self.keys:
            # We fetch logs from the last 24 hours (86400 seconds)
            history = self.db.load_history(self.provider_name, key_usage.api_key, 86400)
            
            for timestamp, tokens in history:
                # We add them to memory so can_make_request works immediately
                key_usage.record_request_in_memory(tokens, timestamp)
            
            # Prune immediately to remove anything that expired between DB load and now
            key_usage.get_current_usage()
    
    def _periodic_cleanup(self):
        """Iterate over all keys and trigger the cleanup/usage calculation."""
        with self.lock:
            for key in self.keys:
                key.get_current_usage()
            
            try:
                self.db.prune_old_records()
            except Exception as e:
                print(f"DB Prune error: {e}")
                
            
        if not self._stop_event.is_set():
            self._cleanup_thread = Timer(self.CLEANUP_INTERVAL_SECONDS, self._periodic_cleanup)
            self._cleanup_thread.daemon = True
            self._cleanup_thread.start()
    
    def _start_cleanup_thread(self):
        """Start the periodic cleanup thread."""
        self._cleanup_thread = Timer(self.CLEANUP_INTERVAL_SECONDS, self._periodic_cleanup)
        self._cleanup_thread.daemon = True # Allows the main program to exit without waiting for the thread
        self._cleanup_thread.start()
    
    def _stop_cleanup_thread(self):
        """Stop the periodic cleanup thread gracefully."""
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.cancel()
    
    def get_available_key(self, estimated_tokens: int = 0) -> Optional[KeyUsage]:
        """Get an available API key that can handle the request"""
        with self.lock:
            for offset in range(len(self.keys)):
                idx = (self.current_key_index + offset) % len(self.keys)
                key = self.keys[idx]
                
                if key.can_make_request(self.rate_limits, estimated_tokens):
                    self.current_key_index = idx
                    return key
            return None
    
    def record_usage(self, api_key: Union[str, KeyUsage], tokens_used: int = 0):
        """Record usage for a specific API key"""
        with self.lock:
            key_identifier = api_key if isinstance(api_key, str) else api_key.api_key
            for key in self.keys:
                if key.api_key == key_identifier:
                    key.record_request_in_memory(tokens_used)
                    self.db.record_usage(self.provider_name, key.api_key, tokens_used)
                    break
    
    def get_stats(self, start: int = 1, end: Optional[int] = None) -> ProviderStats:
        """Get overall statistics as a structured dataclass"""
        
        # Handle 0-based indexing safety
        start_idx = max(start-1, 0)
        end_idx = end if end is not None else len(self.keys)
        
        with self.lock:
            # Slice the keys we want to inspect
            sliced_keys = self.keys[start_idx:end_idx]
            
            # Calculate totals for just the sliced view
            total_requests = sum(k.total_requests for k in sliced_keys)
            total_tokens = sum(k.total_tokens for k in sliced_keys)
            
            key_stats_list = []
            
            # Enumerate using the actual index relative to the full list
            for i, key in enumerate(sliced_keys, start=start_idx):
                usage = key.get_current_usage()
                
                # Create the inner Utilization dataclass
                util_stats = UtilizationStats(
                    rpm=f"{usage['requests_per_minute']}/{self.rate_limits.requests_per_minute}",
                    rph=f"{usage['requests_per_hour']}/{self.rate_limits.requests_per_hour}",
                    rpd=f"{usage['requests_per_day']}/{self.rate_limits.requests_per_day}"
                )
                
                # Create the KeyStats dataclass
                k_stat = KeyStats(
                    key_index=i,
                    key_suffix=key.api_key[-8:] if len(key.api_key) > 8 else '***',
                    current_rpm=usage['requests_per_minute'],
                    current_rph=usage['requests_per_hour'],
                    current_rpd=usage['requests_per_day'],
                    current_tpm=usage['tokens_per_minute'],
                    current_tph=usage['tokens_per_hour'],
                    current_tpd=usage['tokens_per_day'],
                    total_requests=usage['total_requests'],
                    total_tokens=usage['total_tokens'],
                    utilization=util_stats
                )
                key_stats_list.append(k_stat)
            
            # Return the top-level ProviderStats dataclass
            return ProviderStats(
                provider=self.provider_name,
                total_keys=len(self.keys),
                current_key_index=self.current_key_index,
                total_requests=total_requests,
                total_tokens=total_tokens,
                keys=key_stats_list
            )
       
    def wait_for_availability(self, estimated_tokens: int = 0, timeout: float = 10) -> Optional[KeyUsage]:
        """Wait for an available key, with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            key = self.get_available_key(estimated_tokens)
            if key: return key
            time.sleep(0.1)
        
        return None     

class MultiProviderWrapper:
    """Wrapper for Agno models with rotating API keys"""
    
    MODEL_LIMITS = {
        'cerebras': {
            'gpt-oss-120b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=900,
                requests_per_day=14400,
                tokens_per_minute=60000,
                tokens_per_hour=1000000,
                tokens_per_day=1000000,
            ),
            'llama3.1-8b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=900,
                requests_per_day=14400,
                tokens_per_minute=60000,
                tokens_per_hour=1000000,
                tokens_per_day=1000000,
            ),
            'llama-3.3-70b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=900,
                requests_per_day=14400,
                tokens_per_minute=60000,
                tokens_per_hour=1000000,
                tokens_per_day=1000000,
            ),
            'qwen-3-32b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=900,
                requests_per_day=14400,
                tokens_per_minute=60000,
                tokens_per_hour=1000000,
                tokens_per_day=1000000,
            ),
            'qwen-3-235b-a22b-instruct-2507': RateLimits(
                requests_per_minute=30,
                requests_per_hour=900,
                requests_per_day=14400,
                tokens_per_minute=60000,
                tokens_per_hour=1000000,
                tokens_per_day=1000000,
            ),
            'zai-glm-4.6': RateLimits(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=100,
                tokens_per_minute=150000,
                tokens_per_hour=1000000,
                tokens_per_day=1000000,
            ),
        },
        'groq': {
            'allam-2-7b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=7000,
                tokens_per_minute=6000,
                tokens_per_hour=360000,
                tokens_per_day=500000,
            ),
            'groq/compound': RateLimits(
                requests_per_minute=30,
                requests_per_hour=250,
                requests_per_day=250,
                tokens_per_minute=70000,
            ),
            'groq/compound-mini': RateLimits(
                requests_per_minute=30,
                requests_per_hour=250,
                requests_per_day=250,
                tokens_per_minute=70000,
            ),
            'llama-3.1-8b-instant': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=6000,
                tokens_per_hour=360000,
                tokens_per_day=500000,
            ),
            'llama-3.3-70b-versatile': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=12000,
                tokens_per_hour=720000,
                tokens_per_day=100000,
            ),
            'meta-llama/llama-4-maverick-17b-128e-instruct': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=6000,
                tokens_per_hour=360000,
                tokens_per_day=500000,
            ),
            'meta-llama/llama-4-scout-17b-16e-instruct': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=30000,
                tokens_per_hour=1800000,
                tokens_per_day=500000,
            ),
            'meta-llama/llama-guard-4-12b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
                tokens_per_day=500000,
            ),
            'meta-llama/llama-prompt-guard-2-22m': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
                tokens_per_day=500000,
            ),
            'meta-llama/llama-prompt-guard-2-86m': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
                tokens_per_day=500000,
            ),
            'moonshotai/kimi-k2-instruct': RateLimits(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=10000,
                tokens_per_hour=600000,
                tokens_per_day=300000,
            ),
            'moonshotai/kimi-k2-instruct-0905': RateLimits(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=10000,
                tokens_per_hour=600000,
                tokens_per_day=300000,
            ),
            'openai/gpt-oss-120b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=8000,
                tokens_per_hour=480000,
                tokens_per_day=200000,
            ),
            'openai/gpt-oss-20b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=8000,
                tokens_per_hour=480000,
                tokens_per_day=200000,
            ),
            'openai/gpt-oss-safeguard-20b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=8000,
                tokens_per_hour=480000,
                tokens_per_day=200000,
            ),
            'playai-tts': RateLimits(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=100,
                tokens_per_minute=1200,
                tokens_per_hour=72000,
                tokens_per_day=3600,
            ),
            'playai-tts-arabic': RateLimits(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=100,
                tokens_per_minute=1200,
                tokens_per_hour=72000,
                tokens_per_day=3600,
            ),
            'qwen/qwen3-32b': RateLimits(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=1000,
                tokens_per_minute=6000,
                tokens_per_hour=360000,
                tokens_per_day=500000,
            ),
            'whisper-large-v3': RateLimits(
                requests_per_minute=20,
                requests_per_hour=2000,
                requests_per_day=2000,
            ),
            'whisper-large-v3-turbo': RateLimits(
                requests_per_minute=20,
                requests_per_hour=2000,
                requests_per_day=2000,
            ),
        },
        'gemini': {
            'gemini-2.5-flash': RateLimits(
                requests_per_minute=5,
                requests_per_hour=300,
                requests_per_day=20,
                tokens_per_minute=250000,
                tokens_per_hour=15000000,
            ),
            'gemini-2.5-flash-lite': RateLimits(
                requests_per_minute=10,
                requests_per_hour=600,
                requests_per_day=20,
                tokens_per_minute=250000,
                tokens_per_hour=15000000,
            ),
            'gemini-2.5-flash-tts': RateLimits(
                requests_per_minute=3,
                requests_per_hour=180,
                requests_per_day=10,
                tokens_per_minute=10000,
                tokens_per_hour=600000,
            ),
            'gemini-robotics-er-1.5-preview': RateLimits(
                requests_per_minute=10,
                requests_per_hour=600,
                requests_per_day=250,
                tokens_per_minute=250000,
                tokens_per_hour=15000000,
            ),
            'gemma-3-12b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
            ),
            'gemma-3-1b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
            ),
            'gemma-3-27b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
            ),
            'gemma-3-2b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
            ),
            'gemma-3-4b': RateLimits(
                requests_per_minute=30,
                requests_per_hour=1800,
                requests_per_day=14400,
                tokens_per_minute=15000,
                tokens_per_hour=900000,
            ),
        },
    }
    
    @staticmethod
    def load_api_keys_from_env(provider: str, env_file: Optional[str] = None) -> List[str]:
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
            raise ValueError(
                f"Environment variable '{num_keys_var}' not found. "
                f"Please set it in your .env file or environment."
            )
        try:
            num_keys = int(num_keys)
        except ValueError:
            raise ValueError(f"'{num_keys_var}' must be an integer, got: {num_keys}")

        api_keys = []
        for i in range(1, num_keys + 1):
            key_var = f"{provider}_api_key_{i}"
            key = os.getenv(key_var)
            if not key: raise ValueError(f"Missing API key: {provider}_api_key_{i}")
            api_keys.append(key)
        return api_keys
    
    @classmethod
    def from_env(cls, provider: str, model_class: Any, model_id: str, 
                 env_file: Optional[str] = None, custom_limits: Optional[RateLimits] = None,
                 db_path: str = "api_usage.db", **kwargs):
        api_keys = cls.load_api_keys_from_env(provider, env_file)
        return cls(provider, model_class, model_id, api_keys, custom_limits, db_path, **kwargs)
    
    def __init__(self, provider: str, model_class: Any, model_id: str, api_keys: List[str], 
                 custom_limits: Optional[RateLimits] = None, db_path: str = "api_usage.db", **kwargs):
        """
        Initialize wrapper with API keys
        
        Args:
            provider: Provider name ('cerebras', 'groq', 'gemini')
            model_class: The model class (e.g., Cerebras, Groq)
            model_id: Model identifier (e.g., 'llama3.1-8b', 'gemini-1.5-flash')
            api_keys: List of API keys to rotate through
            custom_limits: Optional custom rate limits (overrides model-specific limits)
        """
        self.provider = provider.lower()
        self.model_class = model_class
        self.model_id = model_id
        self.model_kwargs = kwargs
        
        if custom_limits:
            limits = custom_limits
        else:
            provider_models = self.MODEL_LIMITS.get(self.provider, {})
            limits = provider_models.get(model_id)
            if not limits: raise ValueError(f"No rate limits defined for {provider}/{model_id}")
        
        self.key_manager = RotatingKeyManager(api_keys, limits, f"{provider}/{model_id}", db_path)
        self._model_cache = {}

    def _get_model(self, api_key: str):
        """Get or create model instance for API key"""
        if api_key not in self._model_cache:
            model = self.model_class(
                id=self.model_id,
                api_key=api_key,
                **self.model_kwargs
            )
            # monkey-patching
            original_get_metrics = model._get_metrics
            def automated_metrics_hook(*args, **kwargs):
                metrics = original_get_metrics(*args, **kwargs)
                if metrics and hasattr(metrics, 'total_tokens'):
                    self.record_usage(api_key, tokens_used=metrics.total_tokens)
             
            model._get_metrics = automated_metrics_hook
            
            self._model_cache[api_key] = model
            
        return self._model_cache[api_key]
    
    def get_model(self, estimated_tokens: int = 0, wait: bool = True, timeout: float = 10):
        """
        Get a model instance with an available API key
        
        Args:
            estimated_tokens: Estimated tokens for this request
            wait: Whether to wait for availability
            timeout: Maximum wait time in seconds
        
        Returns:
            tuple: (model_instance, api_key)
        """
        if wait:
            key_usage = self.key_manager.wait_for_availability(estimated_tokens, timeout)
        else:
            key_usage = self.key_manager.get_available_key(estimated_tokens)
        
        if not key_usage:
            raise RuntimeError(f"No available API keys for {self.provider} (waited {timeout}s)")
        
        model = self._get_model(key_usage.api_key)
        return model, key_usage

    def record_usage(self, api_key: str, tokens_used: int = 0):
        self.key_manager.record_usage(api_key, tokens_used)
      
    def get_stats(self, start: int = 1, end: Optional[int] = None) -> ProviderStats:
        return self.key_manager.get_stats(start, end)
    
    def print_stats(self, start: int = 1, end: Optional[int] = None):
        """Print formatted statistics using dot notation"""
        stats = self.get_stats(start, end)
        
        print(f"\n{'='*80}")
        print(f"{stats.provider.upper()} Statistics")
        print(f"{'='*80}")
        print(f"Total Keys: {stats.total_keys}")
        print(f"Current Key Index: {stats.current_key_index}")
        print(f"Total Requests: {stats.total_requests:,}")
        print(f"Total Tokens: {stats.total_tokens:,}")
        print(f"\nPer-Key Breakdown:")
        print(f"{'-'*80}")
        
        for key_stat in stats.keys:
            print(f"\nKey #{start} (...{key_stat.key_suffix})")
            print(f"  Current: {key_stat.utilization.rpm} RPM | "
                  f"{key_stat.utilization.rph} RPH | "
                  f"{key_stat.utilization.rpd} RPD")
            print(f"  Tokens: {key_stat.current_tpm:,} TPM | "
                  f"{key_stat.current_tph:,} TPH | "
                  f"{key_stat.current_tpd:,} TPD")
            print(f"  Total: {key_stat.total_requests:,} requests | "
                  f"{key_stat.total_tokens:,} tokens")
        
if __name__ == "__main__":
    
    from agno.models.cerebras import Cerebras
    from agno.models.groq import Groq
    from agno.models.google.gemini import Gemini
    from agno.agent import Agent
    from agno.utils.pprint import pprint_run_response
    
    
    cerebras = MultiProviderWrapper.from_env(
        provider='cerebras',
        model_class=Cerebras,
        model_id='llama3.1-8b',
        max_completion_tokens=512,
        temperature=0.7,
    )

    groq = MultiProviderWrapper.from_env(
        provider='groq',
        model_class=Groq,
        model_id='llama-3.3-70b-versatile'
    )

    gemini = MultiProviderWrapper.from_env(
        provider='gemini',
        model_class=Gemini,
        model_id='gemini-2.5-flash'
    )
    
    model, api_key = cerebras.get_model()
    # model, _ = cerebras.get_model()
    agent = Agent(model=model, markdown=True)
    response = agent.run("Write me a story")
    pprint_run_response(response)
    print(response.metrics.output_tokens)
    cerebras.record_usage(api_key, tokens_used=response.metrics.output_tokens)
    print(api_key.get_current_usage())
    time.sleep(61)
    print(api_key.get_current_usage())
    
    
    
    