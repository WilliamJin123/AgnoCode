from agno.os import AgentOS

from key_manager import MultiProviderWrapper
from companion import WithCompanionAgent
from test_companion import data_worker, companion_system, ENV_FILE, DB_FILE

aOs = AgentOS( 
    id="companion_test_os",
    description="Companion Functionality Test",
    agents=[data_worker],

)
app = aOs.get_app()

if __name__ == "__main__":
    aOs.serve(
        app="os_playground:app",
        port=7777,
        # workers=3,
        reload=True
    )




