import inspect
from typing import Optional, AsyncIterator, Union, List, Dict, Any, Iterator
from agno.agent import Agent, RunOutput, RunOutputEvent
from agno.team import Team
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb
from key_manager import MultiProviderWrapper

import sqlite3
import asyncio
import functools


AgnoWorker = Union[Agent, Team, Workflow]
ENV_FILE="../api_management/.env"
class WithCompanionAgent:
    def __init__(
        self, 
        worker: AgnoWorker, 
        db_path: str = "companion_agent.db",
        instructions: Optional[Union[str, list]] = None,
        model_settings: dict = {"id": "llama-3.3-70b", "provider": "cerebras"},
        router_settings: dict = {"id": "llama-3.1-8b-instant", "provider": "groq"},
        full_context_window: int = 2,  
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
            **kwargs: Additional keyword arguments passed directly to the 
                Companion Agent's constructor (e.g., markdown=True).
        """
       
        self.full_context_buffer: List[Dict[str, str]] = []
        self.full_context_window = full_context_window

        # 1. Initialize SQLite for Summaries
        self.db_path = db_path
        self._init_summary_db()
        self.session_summaries = self._load_summaries()

        self.worker = worker
        p_name = model_settings["provider"].lower()
        r_p_name = router_settings["provider"].lower()
       
        self.model_wrapper = MultiProviderWrapper.from_env(
            provider=p_name,
            default_model_id=model_settings["id"],
            env_file=ENV_FILE
            
        )
        self.router_wrapper = MultiProviderWrapper.from_env(
            provider=r_p_name,
            default_model_id=router_settings["id"],
            env_file=ENV_FILE
        )

        self.model_settings = model_settings 
        self.router_settings = router_settings

        router_model = self.router_wrapper.get_model(**self.router_settings)
        self.router_agent = Agent(model=router_model)

        self.summarizer = Agent(
            model=self.model_wrapper.get_model(**self.model_settings),
            instructions=[
                "You are a technical summarizer.",
                "You will receive the 'existing_summaries' of the project as context.",
                "Summarize the provided User Input and Worker Output into a single concise log entry.",
                "Ensure your new summary flows logically from the existing summaries."
            ]
        )

        base_instructions = [
            "You are a 'Sidecar Companion'. You are observing a conversation between "
            "a User and a specialized AI Worker.",
            "Explain the Worker's reasoning and clarify technical terms.",
            "Do not perform the Worker's task; only explain it."
        ]
        if instructions:
            if isinstance(instructions, str):
                base_instructions.append(instructions)
            else:
                base_instructions.extend(instructions)

        self.companion = Agent(
            model = self.model_wrapper.get_model( **self.model_settings),
            name="Companion Agent",
            id="companion_agent",
            description="A companion agent that answers questions without diluting agentic context",
            instructions=base_instructions,

            db=SqliteDb(db_file=db_path, session_table="Companion Sessions"),
            add_history_to_context=True,
            
            **kwargs,
        )

        self._original_run = self.worker.run
        self._original_arun = self.worker.arun
        self.apply_patches()

    def _init_summary_db(self):
        """Creates the summary table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _load_summaries(self) -> List[str]:
        """Loads all past summaries from the DB."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT summary FROM session_summaries ORDER BY id ASC")
            return [row[0] for row in cursor.fetchall()]

    def _persist_summary(self, summary_text: str):
        """Writes a new summary to the DB."""
        with sqlite3.connect(self.db_path, ) as conn:
            conn.execute("INSERT INTO session_summaries (summary) VALUES (?)", (summary_text,))
            conn.commit()

    def _generate_incremental_summary(self, turn: dict) -> str:
        """
        Compresses a single turn (User Input + Worker Output) into a summary.
        """
        prompt = (
            f"User: {turn['input']}\n"
            f"Worker: {turn['output']}\n"
        )
        deps = {
            "existing_summaries": self.session_summaries
        }

        response = self.summarizer.run(
            prompt,
            dependencies=deps,
            add_dependencies_to_context=True
        )
        summary = str(response.content).strip()
        self._persist_summary(summary)
        return summary

    def _update_worker_state(self, user_input: str, worker_output: str):
        """
        The Rolling Window Logic:
        1. If we have a 'last_turn', summarize it and add to permanent session history.
        2. Set the current interaction as the new 'last_turn' (full fidelity).
        """
        new_turn = {
            "input": user_input,
            "output": worker_output
        }
        self.full_context_buffer.append(new_turn)
        if len(self.full_context_buffer) > self.full_context_window:
            oldest_turn = self.full_context_buffer.pop(0)
            summary = self._generate_incremental_summary(oldest_turn)
            self.session_summaries.append(summary)

    async def _aupdate_worker_state(self, user_input: str, worker_output: str):
        """Async version of state update to prevent blocking on summary generation."""
        new_turn = {
            "input": user_input,
            "output": worker_output
        }
        self.full_context_buffer.append(new_turn)

        if len(self.full_context_buffer) > self.full_context_window:
            oldest_turn = self.full_context_buffer.pop(0)
            summary = await asyncio.to_thread(self._generate_incremental_summary, oldest_turn)
            self.session_summaries.append(summary)

    def _get_companion_dependencies(self) -> Dict[str, Any]:
        """Constructs the dependency dictionary for the Companion."""
        return {
            "summarized_history": self.session_summaries,
            "recent_context_window": self.full_context_buffer
        }

    def _route_query(self, input: str) -> str:
        # Use a concise, high-contrast system prompt
        routing_prompt = (
            "You are a routing logic gate. Categorize the user's message.\n"
            "Respond with ONLY 'WORKER' or 'COMPANION'.\n"
            "WORKER: Execution, Code, Data Analysis.\n"
            "COMPANION: Questions, Definitions, Why, History.\n"
            f"User message: '{input}'"
        )
        
        response = self.router_agent.run(input=routing_prompt)
        decision = str(response.content).strip().upper()

        if "COMPANION" in decision:
            return "COMPANION"
        return "WORKER"
    
    def _patched_run(self, input: str, *args, **kwargs) -> Union[RunOutput, Iterator[Any]]:
        """The synchronous monkeypatch."""
        route = self._route_query(input)
        is_streaming = kwargs.get("stream", False)
        
        if route == "COMPANION":
            deps = self._get_companion_dependencies()

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

    def _stream_interceptor(self, input_text: str, generator: Iterator[Any]) -> Iterator[Any]:
        """
        Yields tokens to the user while secretly buffering them.
        When stream ends, updates the worker state.
        """
        full_response_buffer = []
        
        for chunk in generator:
            token = getattr(chunk, "content", None)
            if token is None and isinstance(chunk, str):
                token = chunk  
            if token:
                full_response_buffer.append(str(token))
            
            yield chunk  # Pass the data through to the user immediately

        # Stream finished: reconstruct full text and update memory
        full_text = "".join(full_response_buffer)
        self._update_worker_state(input_text, full_text)

    async def _patched_arun(self, input: str, *args, **kwargs) -> RunOutput:
        """The asynchronous monkeypatch."""

        route = await asyncio.to_thread(self._route_query, input)
        is_streaming = kwargs.get("stream", False)

        if route == "COMPANION":
            deps = self._get_companion_dependencies()

            return await self.companion.arun(
                input=input,
                dependencies=deps,
                add_dependencies_to_context=True,
                *args, 
                **kwargs
            )
            
        worker_response: Union[RunOutput, AsyncIterator[RunOutputEvent]] = await self._original_arun(input, *args, **kwargs)
        if is_streaming:
            return await self._astream_interceptor(input, worker_response)
        
        await self._aupdate_worker_state(
            input, str(worker_response.content).lower().strip())
        return worker_response

    async def _astream_interceptor(self, input_text: str, generator: AsyncIterator[Any]) -> AsyncIterator[Any]:
        full_response_buffer = []
        
        async for chunk in generator:
            token = getattr(chunk, "content", None)
            if token is None and isinstance(chunk, str):
                token = chunk
            
            if token:
                full_response_buffer.append(str(token))
            
            yield chunk

        full_text = "".join(full_response_buffer)
        await self._aupdate_worker_state(input_text, full_text)

    def apply_patches(self):
        """Explicitly applies patches with functools.wraps logic."""
        
        # Sync Patch
        @functools.wraps(self.worker.run)
        def wrapped_run(*args, **kwargs):
            return self._patched_run(*args, **kwargs)
        self.worker.run = wrapped_run

        # Async Patch (Critical for AgentOS)
        @functools.wraps(self.worker.arun)
        async def wrapped_arun(*args, **kwargs):
            return await self._patched_arun(*args, **kwargs)

        self.worker.arun = wrapped_arun
        
