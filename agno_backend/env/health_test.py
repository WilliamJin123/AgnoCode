import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.cerebras import Cerebras
from agno.models.google.gemini import Gemini

load_dotenv(".env", override=True)

def test_provider_keys(prefix, provider_class, model_id, prompt="Say Hello."):
    num = int(os.getenv(f"NUM_{prefix}", 0))
    success = []
    fail = []
    for i in range(1, num + 1):
        key_env = f"{prefix}_API_KEY_{i}"
        key = os.getenv(key_env)
        if not key:
            fail.append(key_env)
            continue
        try:
            agent = Agent(model=provider_class(id=model_id, api_key=key), markdown=True)
            agent.print_response(prompt)
            success.append(key_env)
        except Exception:
            fail.append(key_env)
    print(f"{prefix} successful:", success)
    print(f"{prefix} failed:", fail)
    input("Press Enter to continue...")

# test_provider_keys("GROQ", Groq, "llama-3.3-70b-versatile")
# test_provider_keys("CEREBRAS", Cerebras, "llama3.1-8b")
test_provider_keys("GEMINI", Gemini, "gemini-2.5-flash")
