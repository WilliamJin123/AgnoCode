import inspect
import os
import re
from typing import AsyncGenerator, Optional, AsyncIterator, Union, List, Dict, Any, Iterator
from agno.agent import Agent, RunOutput, RunOutputEvent
from agno.team import Team
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing
from key_manager import MultiProviderWrapper
from tenacity import retry, stop_after_attempt, wait_exponential

import asyncio
import functools
from functools import lru_cache
from sqlalchemy import (
    create_engine, 
    text, 
    MetaData, 
    Table, 
    Column, 
    Integer, 
    String, 
    DateTime,
    select,
    insert,
    func
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager


AgnoWorker = Union[Agent, Team, Workflow]
ENV_FILE="../api_management/.env"
class WithCompanionAgent:
    def __init__(
        self, 
        worker: AgnoWorker, 
        db_path: str = "companion_agent.db",
        instructions: Optional[Union[str, list]] = None,
        model_settings: dict = {"id": "qwen-3-235b-a22b-instruct-2507", "provider": "cerebras"},
        router_settings: dict = {"id": "qwen-3-32b", "provider": "cerebras"},
        full_context_window: int = 2, 
        model_wrapper: Optional[MultiProviderWrapper] = None,
        router_wrapper: Optional[MultiProviderWrapper] = None, 
        **kwargs
    ):
        """
        Wraps an Agno Worker (Agent, Team, or Workflow) with a Follow-Along Sidecar Companion.

        This class acts as a router and coordinator, ensuring that meta-queries or educational 
        questions are handled by the Companion Agent to prevent context dilution in the 
        primary Worker.

        Args:
            worker (AgnoWorker): The primary task-oriented Agent, Team, or Workflow.
            db_path (str): File path for the SQLite database used to persist 
                Companion session history. Defaults to "companion_agent.db".
            instructions (Optional[Union[str, list]]): Additional guidelines to extend 
                the Companion's base sidecar persona.
            model_settings (dict): Configuration for the Companion's LLM. 
                Must include 'id' and 'provider'. 
                Defaults to {"id": "llama-3.3-70b-versatile", "provider": "groq"}.
            router_settings (dict): Configuration for the high-speed Router LLM.
                Must include 'id' and 'provider'.
                Defaults to {"id": "llama-3.1-8b-instant", "provider": "groq"}.
            model_wrapper (Optional[MultiProviderWrapper]): Pre-initialized wrapper for the main model.
            router_wrapper (Optional[MultiProviderWrapper]): Pre-initialized wrapper for the router.
            **kwargs: Additional keyword arguments passed directly to the 
                Companion Agent's constructor (e.g., markdown=True).
        """
        self.worker = worker

        self.full_context_buffer: List[Dict[str, str]] = []
        self.full_context_window = full_context_window

        # 1. Initialize SQLite for Summaries
        self.db_path = os.path.abspath(db_path)
        
        self._init_sqlalchemy_components()
        self._init_summary_db()

        self.model_settings = model_settings 
        self.router_settings = router_settings
        if model_wrapper:
            self.model_wrapper = model_wrapper
        else:
            p_name = model_settings["provider"].lower()
            self.model_wrapper = MultiProviderWrapper.from_env(
                provider=p_name,
                default_model_id=model_settings["id"],
                env_file=ENV_FILE
            )
        if router_wrapper:
            self.router_wrapper = router_wrapper
        else:
            r_p_name = router_settings["provider"].lower()
            self.router_wrapper = MultiProviderWrapper.from_env(
                provider=r_p_name,
                default_model_id=router_settings["id"],
                env_file=ENV_FILE
            )

        router_model = self.router_wrapper.get_model(**self.router_settings)
        self.router_agent = Agent(model=router_model)

        self.summarizer = Agent(
            model=self.model_wrapper.get_model(**self.model_settings),
            instructions=[
                "You are a technical summarizer.",
                "You will receive the 'past_summaries' of the project as context.",
                "Summarize the provided User Input and Worker Output into a single concise log entry.",
                "Ensure your new summary flows logically from the existing summaries."
            ]
        )

        base_instructions = [
            "ACT AS THE WORKER: You are a seamless extension of the primary AI Worker. ",
            "Answer conceptual or clarifying questions about the task at hand "
            "so the primary Worker's context remains focused on execution.",
            "BE TRANSPARENT BUT INVISIBLE: Do not explain that you are a sidecar or a companion. "
            "Just provide the clarification as if the Worker itself is answering a quick tangent."
        ]
        if instructions:
            if isinstance(instructions, str):
                base_instructions.append(instructions)
            else:
                base_instructions.extend(instructions)

        self.db = SqliteDb(db_file=db_path, session_table="companion_sessions", traces_table="companion_traces")
        setup_tracing(
            db=self.db,
            batch_processing=True,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000,
        )
        self.companion = Agent(
            model = self.model_wrapper.get_model( **self.model_settings),
            name="Companion Agent",
            id="companion_agent",
            description="A companion agent that answers questions without diluting agentic context",
            instructions=base_instructions,

            db=self.db,
            add_history_to_context=True,
            
            **kwargs,
        )

        self._original_run = self.worker.run
        self._original_arun = self.worker.arun
        self.apply_patches()

    def _init_sqlalchemy_components(self):
        """Initialize SQLAlchemy engine, metadata, and session factory once."""
        self.db_path = os.path.abspath(self.db_path)
        
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False  # SQL logging
        )
        
        # Create metadata and table definition
        self.metadata = MetaData()
        self.session_summaries_table = Table(
            'session_summaries',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('session_id', String, nullable=False, index=True),
            Column('summary', String, nullable=False),
            Column('timestamp', DateTime, server_default=func.now())
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _init_summary_db(self):
        """Creates the summary table if it doesn't exist."""
        self.metadata.create_all(self.engine)

    @contextmanager
    def _get_db_session(self):
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @lru_cache(maxsize=128)
    def _load_summaries(self, session_id: str) -> List[str]:
        """Loads all past summaries from the DB."""
        with self._get_db_session() as session:
            result = session.execute(
                select(self.session_summaries_table.c.summary)
                .where(self.session_summaries_table.c.session_id == session_id)
                .order_by(self.session_summaries_table.c.id)
            )
            return [row[0] for row in result.fetchall()]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _persist_summary(self, session_id: str, summary_text: str):
        """Writes a new summary to the DB."""
        with self._get_db_session() as session:
            session.execute(
                insert(self.session_summaries_table).values(
                    session_id=session_id,
                    summary=summary_text
                )
            )
            self._load_summaries.cache_clear()

    def _generate_incremental_summary(self, turn: dict, session_id: str) -> str:
        """
        Compresses a single turn (User Input + Worker Output) into a summary.
        """
        prompt = (
            f"User: {turn['input']}\n"
            f"Worker: {turn['output']}\n"
        )
        session_summaries = self._load_summaries(session_id)

        deps = {
            "past_summaries": session_summaries
        }

        response = self.summarizer.run(
            prompt,
            dependencies=deps,
            add_dependencies_to_context=True
        )
        summary = str(response.content).strip()
        self._persist_summary(session_id, summary)
        return summary

    def _update_worker_state(self, user_input: str, worker_output: str):
        """
        The Rolling Window Logic:
        1. If we have a 'last_turn', summarize it and add to permanent session history.
        2. Set the current interaction as the new 'last_turn' (full fidelity).
        """
        session_id = getattr(self.worker, "session_id", "default_session")
        new_turn = {
            "input": user_input,
            "output": worker_output
        }
        self.full_context_buffer.append(new_turn)
        if len(self.full_context_buffer) > self.full_context_window:
            oldest_turn = self.full_context_buffer.pop(0)
            _ = self._generate_incremental_summary(oldest_turn, session_id)
            

    async def _aupdate_worker_state(self, user_input: str, worker_output: str):
        """Async version of state update to prevent blocking on summary generation."""
        session_id = getattr(self.worker, "session_id", "default_session")
        new_turn = {
            "input": user_input,
            "output": worker_output
        }
        self.full_context_buffer.append(new_turn)

        if len(self.full_context_buffer) > self.full_context_window:
            oldest_turn = self.full_context_buffer.pop(0)
            _ = await asyncio.to_thread(self._generate_incremental_summary, oldest_turn, session_id)

    def _get_companion_dependencies(self) -> Dict[str, Any]:
        """Constructs the dependency dictionary for the Companion."""
        session_id = getattr(self.worker, "session_id", "default_session")
        return {
            "summarized_history": self._load_summaries(session_id),
            "recent_context_window": self.full_context_buffer
        }

    def _route_query(self, _input: str) -> str:
        # Use a concise, high-contrast system prompt
        routing_prompt = (
            "Analyze the User message to determine the intended recipient.\n"
            "Respond with ONLY 'WORKER' or 'COMPANION'.\n\n"
            
            "WORKER: Use if the user is giving instructions, correcting code, "
            "requesting changes, or challenging a specific technical decision "
            "made in the current task (e.g., 'Use X pattern instead').\n"
            
            "COMPANION: Use ONLY for general technical questions, definitions, "
            "educational 'how-to' explanations, or history that does NOT "
            "require the Worker to change its current output or logic.\n\n"
            
            f"User message: '{_input}'"
        )
        
        response = self.router_agent.run(input=routing_prompt)
        content = str(response.content)
        decision = re.sub(
            r'<think>.*?</think>', 
            '', 
            content, 
            flags=re.DOTALL | re.IGNORECASE
        ).strip().upper()
        if "COMPANION" in decision:
            return "COMPANION"
        return "WORKER"
    
    def _patched_run(self, input: str, *args, **kwargs) -> Union[RunOutput, Iterator[Any]]:
        """The synchronous monkeypatch."""
        route = self._route_query(input)
        is_streaming = kwargs.get("stream", False)
        
        if route == "COMPANION":
            deps = self._get_companion_dependencies()

            kwargs.pop('dependencies', None)
            kwargs.pop('add_dependencies_to_context', None)

            return self.companion.run(
                input=input, 
                dependencies=deps, 
                add_dependencies_to_context=True,
                *args, 
                **kwargs
            )

        response = self._original_run(input, *args, **kwargs)

        if is_streaming:
            return self._stream_interceptor(input, response)
        
        self._update_worker_state(
            input, str(response.content).lower().strip())
        return response

    async def _resolve_routing(self, input_text: str):
        """
        Decides whether to route to Companion or Worker.
        Returns: (route_decision, clean_dependencies_flag)
        """
        route = await asyncio.to_thread(self._route_query, input_text)
        return route

    async def _arun_stream(self, input: str, *args, **kwargs) -> AsyncGenerator:

        route = await self._resolve_routing(input)
        if route == "COMPANION":
            deps = self._get_companion_dependencies()
            kwargs.pop('dependencies', None) # sidecar does not get worker-specific dependencies
            
            stream: AsyncGenerator = self.companion.arun(
                input=input, dependencies=deps, *args, **kwargs
            )
            async for chunk in stream:
                yield chunk
            return
        
        result: AsyncGenerator = self._original_arun(input, *args, **kwargs)

        input_text = kwargs.get('input') or input

        async for chunk in self._astream_interceptor(input_text, result):
            yield chunk

    async def _arun_standard(self, input: str, *args, **kwargs) -> Any:

        route = await self._resolve_routing(input)

        if route == "COMPANION":
            deps = self._get_companion_dependencies()
            kwargs.pop('dependencies', None)
            kwargs.pop("stream", None)

            return await self.companion.arun(
                input=input, dependencies=deps, stream=False, *args, **kwargs
            )

        # Delegate to Worker's standard run
        result = self._original_arun(input, *args, **kwargs)
        if inspect.isawaitable(result):
            result = await result

        # Log state
        input_text = kwargs.get('input') or input
        asyncio.create_task(self._aupdate_worker_state(input_text, str(result.content).lower().strip()))
        
        return result

    def _stream_interceptor(self, input_text: str, generator: Iterator[Any]) -> Iterator[Any]:
        """
        Yields tokens to the user while secretly buffering them.
        When stream ends, updates the worker state.
        """
        full_response_buffer = []
        
        for chunk in generator:
            token = None
            if hasattr(chunk, "content") and chunk.content is not None:
                token = chunk.content
            elif isinstance(chunk, str):
                token = chunk     
            if token:
                full_response_buffer.append(str(token))
            yield chunk

        full_text = "".join(full_response_buffer)
        self._update_worker_state(input_text, full_text)

    async def _astream_interceptor(self, input_text: str, generator:    AsyncIterator[Any]) -> AsyncIterator[Any]:
        full_response_buffer = []
        
        async for chunk in generator:
            token = None
            if hasattr(chunk, "content") and chunk.content is not None:
                token = chunk.content
            elif isinstance(chunk, str):
                token = chunk     
            if token:
                full_response_buffer.append(str(token))
            yield chunk
        
        full_text = "".join(full_response_buffer)
        asyncio.create_task(self._aupdate_worker_state(input_text, full_text))

    def apply_patches(self):
        """Explicitly applies patches with functools.wraps logic."""
        
        # Sync Patch
        @functools.wraps(self.worker.run)
        def wrapped_run(*args, **kwargs):
            return self._patched_run(*args, **kwargs)
        self.worker.run = wrapped_run

        # Async Patch (Critical for AgentOS)
        @functools.wraps(self.worker.arun)
        def arun_dispatch(agent_self, *args, **kwargs):
            input_val = kwargs.pop("input", None)
            if input_val is None and args:
                input_val = args[0]
                args = args[1:]
            if kwargs.get("stream", False):
                return self._arun_stream(input_val, *args, **kwargs)
            else:
                return self._arun_standard(input_val, *args, **kwargs)
        self.worker.arun = arun_dispatch.__get__(self.worker, type(self.worker))
        
