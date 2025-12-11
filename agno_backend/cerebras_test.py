import asyncio
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from key_manager import MultiProviderWrapper
from agno.agent import Agent
from agno.models.cerebras import Cerebras

ENV_FILE = "../env/.env"
DB_FILE = "../env/api_usage.db"
MAX_TOKENS = 1000
cerebras_wrapper = MultiProviderWrapper.from_env(
        provider="cerebras",  # Using string (validated inside) or Provider.CEREBRAS
        model_class=Cerebras, 
        default_model_id='zai-glm-4.6', 
        env_file=ENV_FILE,
        db_path=DB_FILE,
        # max_completion_tokens=MAX_TOKENS  # Kwarg passed to model
    )

load_dotenv(ENV_FILE)

if __name__ == "__main__":
    
    agent1 = Agent(model=cerebras_wrapper.get_model()[0])
    agent2 = Agent(model=Cerebras(id="zai-glm-4.6",api_key=os.getenv("CEREBRAS_API_KEY_10")))
    
    prompt = "Explain the theory of relativity in simple terms."
    
    for agent in [agent1, agent2]:
        print(f"\n[AGENT USING KEY: ...{agent.model.api_key[-8:]}] SENDING PROMPT")
        response = agent.run(prompt)
        print(f"Response: {response.content}\n")
        input("Press Enter to continue to next agent...")
    