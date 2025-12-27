
from ..config.dataclasses import KeyUsage
from ..config.log_config import default_logger

class RotatingCredentialsMixin:
    """
    Mixin that handles key rotation, 429 detection, and 30s cooldown triggers.
    """
    
    def __init__(self, *args, logger=None, **kwargs):
        self.logger = logger or default_logger
        super().__init__(*args, **kwargs)

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
                    self.logger.warning("429 Hit on key %s (Sync). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
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
                    self.logger.warning("429 Hit on key %s (Async). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
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
                    self.logger.warning("429 Hit on key %s (Sync Stream). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
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
                    self.logger.warning("429 Hit on key %s (Async Stream). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e
