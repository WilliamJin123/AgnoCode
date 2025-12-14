import os
import requests
import json
from typing import Optional, Dict, List, Any
from agno.tools import Toolkit

class LinearTools(Toolkit):
    def __init__(self, api_key: str = None):
        super().__init__(name="linear_tools")
        self.api_key = api_key or os.getenv("LINEAR_API_KEY")
        self.url = "https://api.linear.app/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key
        }
        
        # Register Tools
        self.register(self.get_teams)
        # self.register(self.ensure_label)
        self.register(self.ensure_labels)
        self.register(self.get_issues)
        self.register(self.batch_create_issues)
        self.register(self.batch_update_issues)
        self.register(self.batch_delete_issues)

    def _query(self, query: str, variables: dict = {}) -> dict:
        """Internal helper to send GraphQL requests."""
        response = requests.post(
            self.url, 
            headers=self.headers, 
            json={"query": query, "variables": variables}
        )
        if response.status_code != 200:
            raise Exception(f"Linear API Error: {response.text}")
        
        data = response.json()
        if "errors" in data:
            raise Exception(f"GraphQL Error: {data['errors'][0]['message']}")
        return data["data"]

    # --- READ / SETUP OPERATIONS ---

    def get_teams(self) -> str:
        """Lists teams and their workflow states."""
        query = """
        query {
          teams {
            nodes {
              id name key
              states { nodes { id name type } }
            }
          }
        }
        """
        try:
            data = self._query(query)
            teams = []
            for team in data['teams']['nodes']:
                states = {s['name']: {'id': s['id'], 'type': s['type']} for s in team['states']['nodes']}
                teams.append({"name": team['name'], "id": team['id'], "states": states})
            return json.dumps(teams, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_issues(self, team_id: str) -> str:
        """
        Fetches active issues with priority to calculate the 'Diff'.
        """
        query = """
        query($teamId: String!) {
          team(id: $teamId) {
            issues(first: 100, filter: { state: { type: { neq: "completed" } } }) {
              nodes { 
                id 
                title 
                description 
                priority 
                state { id name } 
              }
            }
          }
        }
        """
        try:
            data = self._query(query, {"teamId": team_id})
            issues = []
            for node in data['team']['issues']['nodes']:
                issues.append({
                    "id": node['id'],
                    "title": node['title'],
                    "priority": node['priority'], # 0=None, 1=Urgent, 2=High...
                    "state_id": node['state']['id'],
                    "state_name": node['state']['name']
                })
            return json.dumps(issues, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

    def ensure_label(self, team_id: str, label_name: str) -> str:
        """Ensures a label exists and returns its ID."""
        search_query = 'query($name: String!) { issueLabels(filter: { name: { eq: $name } }) { nodes { id } } }'
        data = self._query(search_query, {"name": label_name})
        if data['issueLabels']['nodes']: return data['issueLabels']['nodes'][0]['id']
        
        create_mutation = 'mutation($name: String!, $teamId: String!) { issueLabelCreate(input: { name: $name, teamId: $teamId }) { issueLabel { id } } }'
        return self._query(create_mutation, {"name": label_name, "teamId": team_id})['issueLabelCreate']['issueLabel']['id']

    def ensure_labels(self, team_id: str, label_names: List[str]) -> str:
        """
        Ensures multiple labels exist in Linear. Creates any that are missing.
        Returns a map of Name -> UUID for all requested labels.
        
        Args:
            team_id (str): The Team ID.
            label_names (List[str]): List of label names to ensure (e.g. ["Bug", "Feature"]).
        """
        try:
            # 1. Fetch ALL existing labels for the team to avoid duplicates
            # We fetch first to save write operations
            query = """
            query($teamId: String!) {
              team(id: $teamId) {
                labels(first: 100) {
                  nodes { id name }
                }
              }
            }
            """
            data = self._query(query, {"teamId": team_id})
            
            # Map existing Name -> ID
            label_map = {
                node['name']: node['id'] 
                for node in data['team']['labels']['nodes']
            }

            # 2. Identify which ones are missing
            missing_labels = [name for name in label_names if name not in label_map]

            if not missing_labels:
                # If all exist, just return the map for the requested labels
                result = {name: label_map[name] for name in label_names if name in label_map}
                return json.dumps(result, indent=2)

            # 3. Batch Create Missing Labels using GraphQL Aliases
            mutation_parts = []
            variables = {"teamId": team_id}
            
            for idx, name in enumerate(missing_labels):
                alias = f"create{idx}"
                variables[f"name{idx}"] = name
                mutation_parts.append(f"""
                  {alias}: issueLabelCreate(input: {{ name: $name{idx}, teamId: $teamId }}) {{
                    issueLabel {{ id name }}
                  }}
                """)
            
            # Build query inputs: $name0: String!, $name1: String!...
            var_defs = ["$teamId: String!"] + [f"$name{i}: String!" for i in range(len(missing_labels))]
            full_query = f"mutation({', '.join(var_defs)}) {{\n" + "\n".join(mutation_parts) + "\n}"
            
            # Execute Batch Creation
            create_data = self._query(full_query, variables)
            
            # 4. Update the map with the newly created IDs
            for idx, name in enumerate(missing_labels):
                new_id = create_data[f"create{idx}"]['issueLabel']['id']
                label_map[name] = new_id
            
            # Return only the requested labels
            final_result = {name: label_map[name] for name in label_names}
            return json.dumps(final_result, indent=2)

        except Exception as e:
            return f"Error ensuring labels: {str(e)}"

    # --- BATCH OPERATIONS ---

    def batch_create_issues(self, team_id: str, tasks: str) -> str:
        """
        Creates multiple issues in one request.
        
        Args:
            team_id (str): The Team ID.
            tasks (str): JSON STRING list of task objects.
                         Format: [{"title": "A", "priority": 1, ...}]
                         Priority Map: 0=None, 1=Urgent, 2=High, 3=Medium, 4=Low.
        """
        try:
            task_list = json.loads(tasks)
            if not task_list: return "No tasks to create."

            mutation_parts = []
            variables = {"teamId": team_id}
            
            for idx, task in enumerate(task_list):
                alias = f"create{idx}"
                
                # Dynamic Variables
                variables[f"title{idx}"] = task['title']
                variables[f"desc{idx}"] = task.get('description', '')
                variables[f"state{idx}"] = task.get('state_id')
                variables[f"prio{idx}"] = int(task.get('priority', 0)) # Default to 0
                variables[f"labels{idx}"] = task.get('label_ids', [])
                
                mutation_parts.append(f"""
                  {alias}: issueCreate(input: {{
                    teamId: $teamId
                    title: $title{idx}
                    description: $desc{idx}
                    stateId: $state{idx}
                    priority: $prio{idx}
                    labelIds: $labels{idx}
                  }}) {{ issue {{ id identifier }} }}
                """)

            # Construct query signature
            # Added $prio{i}: Int
            var_defs = ["$teamId: String!"]
            for i in range(len(task_list)):
                var_defs.extend([
                    f"$title{i}: String!", 
                    f"$desc{i}: String", 
                    f"$state{i}: String", 
                    f"$prio{i}: Int", 
                    f"$labels{i}: [String!]"
                ])
            
            full_query = f"mutation({', '.join(var_defs)}) {{\n" + "\n".join(mutation_parts) + "\n}"

            self._query(full_query, variables)
            return f"Successfully batch created {len(task_list)} issues."

        except Exception as e:
            return f"Batch Create Failed: {str(e)}"

    def batch_update_issues(self, updates: str) -> str:
        """
        Updates multiple issues.
        
        Args:
            updates (str): JSON STRING list of update objects.
                           Format: [{"id": "UUID", "priority": 2, "state_id": "..."}]
        """
        try:
            update_list = json.loads(updates)
            if not update_list: return "No updates needed."

            mutation_parts = []
            variables = {}
            
            for idx, update in enumerate(update_list):
                alias = f"update{idx}"
                issue_id = update.pop("id")
                
                # The 'input' variable passed to Linear handles all fields automatically
                # including 'priority' if present in the dictionary.
                variables[f"id{idx}"] = issue_id
                variables[f"input{idx}"] = update 
                
                mutation_parts.append(f"""
                  {alias}: issueUpdate(id: $id{idx}, input: $input{idx}) {{ issue {{ id }} }}
                """)

            var_defs = ", ".join([f"$id{i}: String!, $input{i}: IssueUpdateInput!" for i in range(len(update_list))])
            full_query = f"mutation({var_defs}) {{\n" + "\n".join(mutation_parts) + "\n}"
            
            self._query(full_query, variables)
            return f"Successfully batch updated {len(update_list)} issues."

        except Exception as e:
            return f"Batch Update Failed: {str(e)}"

    def batch_delete_issues(self, issue_ids: str) -> str:
        """Deletes multiple issues."""
        try:
            ids = json.loads(issue_ids)
            if not ids: return "No issues to delete."
            
            mutation_parts = []
            variables = {}

            for idx, i_id in enumerate(ids):
                alias = f"del{idx}"
                variables[f"id{idx}"] = i_id
                mutation_parts.append(f"{alias}: issueDelete(id: $id{idx}) {{ success }}")

            var_defs = ", ".join([f"$id{i}: String!" for i in range(len(ids))])
            full_query = f"mutation({var_defs}) {{\n" + "\n".join(mutation_parts) + "\n}"
            
            self._query(full_query, variables)
            return f"Successfully batch deleted {len(ids)} issues."
        except Exception as e:
            return f"Batch Delete Failed: {str(e)}"