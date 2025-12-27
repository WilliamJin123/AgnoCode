import os
import time
from typing import Optional, List
from sqlalchemy import (
    create_engine, select, and_, Table, Column,
    Integer, String, Float, MetaData, Index, delete,
    URL
)


# --- DATABASE LAYER ---

class UsageDatabase:
    """Handles Online persistence for API usage"""
    def __init__(self, db_url: Optional[str] = None, db_env_var: str = "TIDB_DB_URL"):
        self.db_url = db_url or os.getenv(db_env_var)
        if not self.db_url:
            raise ValueError(f"Database URL not provided and {db_env_var} not set.")
        self.engine = create_engine(
            self.db_url,
            pool_recycle = 300   
        )
        self._init_db()

    def _init_db(self):
        metadata = MetaData()
        self.usage_logs = Table(
            'usage_logs',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('provider', String(100)),
            Column('model', String(100)),
            Column('api_key_suffix', String(50)),
            Column('timestamp', Float),
            Column('tokens', Integer),

            Index('idx_key_usage', 'provider', 'api_key_suffix', 'timestamp'),
            Index('idx_cleanup', 'timestamp'),
            Index('idx_model_reporting', 'provider', 'model', 'timestamp')
        )
        metadata.create_all(self.engine)

    def load_history(self, provider: str, api_key: str, seconds_lookback: int) -> List[tuple[str, float, int]]:
        """Load history SPECIFIC to this Provider + Model combination"""
        suffix = api_key[-8:] if len(api_key) > 8 else api_key
        cutoff = time.time() - seconds_lookback

        stmt = (
            select(
                self.usage_logs.c.model,
                self.usage_logs.c.timestamp,
                self.usage_logs.c.tokens,
            )
            .where(
                and_(
                    self.usage_logs.c.provider == provider,
                    self.usage_logs.c.api_key_suffix == suffix,
                    self.usage_logs.c.timestamp > cutoff,
                )
            )
            .order_by(self.usage_logs.c.timestamp.asc())
        )

        with self.engine.connect() as conn:
            return conn.execute(stmt).all()

    def load_provider_history(self, provider: str, seconds_lookback: int):
        """Optimization: Load everything for the provider in ONE call"""
        cutoff = time.time() - seconds_lookback

        stmt = (
            select(
                self.usage_logs.c.api_key_suffix,
                self.usage_logs.c.model,
                self.usage_logs.c.timestamp,
                self.usage_logs.c.tokens,
            )
            .where(
                self.usage_logs.c.provider == provider,
                self.usage_logs.c.timestamp > cutoff,
            )
        )

        with self.engine.connect() as conn:
            return conn.execute(stmt).all()

    def prune_old_records(self, days_retention: int = 3):
        """Delete records older than retention period to keep DB small (3 days)"""
        cutoff = time.time() - (days_retention * 86400)
        with self.engine.connect() as conn:
            conn.execute(
                delete(self.usage_logs).where(
                self.usage_logs.c.timestamp < cutoff
            ))
            conn.commit()
