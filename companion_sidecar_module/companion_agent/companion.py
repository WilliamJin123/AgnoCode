from typing import Optional, AsyncIterator, Union, List, Dict, Any, Iterator
from agno.agent import Agent, RunResponse
from agno.team import Team
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb
from key_manager import MultiProviderWrapper

import asyncio


AgnoWorker = Union[Agent, Team, Workflow]
ENV_FILE="../api_management/.env"
class WithCompanionAgent:
    def __init__(
        self, 
        worker: AgnoWorker, 
        db_path: str = "companion_agent.db",
        instructions: Optional[Union[str, list]] = None,
        model_settings: dict = {"id": "llama-3.3-70b-versatile", "provider": "groq"},
        router_settings: dict = {"id": "llama-3.1-8b-instant", "provider": "groq"},
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
        self.worker =  worker
        p_name = model_settings["provider"].lower()
        r_p_name = router_settings["provider"].lower()
       
        self.model_wrapper = MultiProviderWrapper.from_env(
            provider=p_name,
            default_model_id=model_settings["id"],
            env_file=ENV_FILE
            
        )
        self.router_wrapper = MultiProviderWrapper(
            provider=r_p_name,
            default_model_id=router_settings["id"],
            env_file=ENV_FILE
        )

        self.model_settings = model_settings or {}

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