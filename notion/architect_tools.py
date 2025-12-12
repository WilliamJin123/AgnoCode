import os
import json
from notion_client import Client
from agno.tools import Toolkit

class NotionKanbanTools(Toolkit):
    def __init__(self, database_id: str, api_key: str = None, **kwargs):
        
        self.client = Client(auth=api_key or os.getenv("NOTION_API_KEY"))
        self.database_id = database_id
        tools = [
            self.get_board_snapshot,
            self.create_card,
            self.create_card,
            self.archive_card
        ]

        super().__init__(name="notion_kanban", **kwargs)

    # pre-hook
    def get_board_snapshot(self) -> str:
        """
        Returns a JSON summary of all active cards (ID, Title, Status).
        """
        results = []
        has_more = True
        next_cursor = None

        while has_more:
            try:
                response = self.client.databases.query(
                    database_id=self.database_id,
                    start_cursor=next_cursor,
                    filter={"property": "Status", "status": {"does_not_equal": "Done"}}
                )
                for page in response['results']:
                    # Safe extraction of Title
                    title_prop = page['properties'].get('Name', {}).get('title', [])
                    title = title_prop[0]['text']['content'] if title_prop else "Untitled"
                    
                    # Safe extraction of Status
                    status_prop = page['properties'].get('Status', {}).get('status', {})
                    status = status_prop.get('name', 'Unknown')
                    
                    results.append(f"ID: {page['id']} | Title: {title} | Status: {status}")
                
                has_more = response['has_more']
                next_cursor = response['next_cursor']
            except Exception:
                has_more = False # Stop on error
            
        return json.dumps(results, indent=2) if results else "Board is empty."
            