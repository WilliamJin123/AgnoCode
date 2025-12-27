import os
import time
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.markup import escape

from .utils import get_agno_model_class
from .key_rotation.rotation_manager import RotatingKeyManager
from .key_rotation.rotating_mixin import RotatingCredentialsMixin
from .config.dataclasses import RateLimits, UsageSnapshot
from .config.enums import RateLimitStrategy
from .usage.db_logic import UsageDatabase
from .config.log_config import default_logger


   

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
            # if not env_path.exists():
                 # print(f"Warning: The provided env_file '{env_path}' does not exist.")
            #     logger.warning("The provided env_file '%s' does not exist.", env_path)
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
        db_env_var: Optional[str] = "TIDB_DB_URL",
        debug: bool = False, 
        logger = None,
        **kwargs):
        
        model_class = get_agno_model_class(provider)
        api_keys = cls.load_api_keys(provider, env_file)
        db_url = db_url or os.getenv("TURSO_DATABASE_URL")
        return cls(provider, model_class, default_model_id, 
                   api_keys, db_url, db_env_var, debug, logger, **kwargs)
    
    def __init__(self, provider: str, model_class: Any, 
                default_model_id: str, api_keys: List[str], 
                db_url: Optional[str] = None, 
                db_env_var: Optional[str] = "TIDB_DB_URL",
                debug: bool = False, logger = None,
                **kwargs):
        self.provider = provider.lower()
        self.logger = logger or default_logger
        self.model_class = model_class
        self.default_model_id = default_model_id
        self.model_kwargs = kwargs
        self.toggle_debug(debug)
        self.db = UsageDatabase(db_url, db_env_var)
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
        self.logger.setLevel(level)
        
        status = "ENABLED" if enable else "DISABLED"
        self.logger.info(f"Debug logging {status} for {self.provider}")

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