import json
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from anthropic import Anthropic
import os

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Constants
BATCH_SIZE = 10  # Process 10 tickets at a time
MAX_WORKERS = 5  # Maximum number of parallel API calls

class Agent(BaseModel):
    """Model representing a support agent."""
    agent_id: str
    name: str
    skills: dict[str, int]
    current_load: int
    availability_status: str
    experience_level: int

class Ticket(BaseModel):
    """Model representing a support ticket."""
    ticket_id: str
    title: str
    description: str
    creation_timestamp: int

class TicketAssignment(BaseModel):
    """Model representing a ticket assignment."""
    ticket_id: str
    title: str
    assigned_agent_id: str
    rationale: str

class TicketProcessor:
    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.model = model
        
    def process_ticket_batch(self, tickets):
        """Process a batch of tickets together."""
        tickets_json = [{"title": t.title, "description": t.description} for t in tickets]
        prompt = f"""Analyze these support tickets and extract key information for each:
        Tickets: {json.dumps(tickets_json)}
        
        For each ticket, extract and return:
        1. Required skills (list of skills needed)
        2. Priority level (1-5, where 5 is highest)
        3. Complexity level (1-5, where 5 is most complex)
        4. Is business critical (true/false)
        
        Return a JSON array with an entry for each ticket, maintaining the same order.
        Return the response in JSON format only, no additional text."""
        
        response = anthropic.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content[0].text)
        
    def extract_ticket_info(self, ticket):
        """Extract key information from a single ticket."""
        return self.process_ticket_batch([ticket])[0]

class SkillMatcher:
    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.model = model
    
    def process_batch(self, ticket_infos, agents, top_k = 3):
        """Find best agents for a batch of tickets."""
        agents_json = [agent.model_dump() for agent in agents]
        prompt = f"""Given these agents and ticket requirements, rank the top {top_k} most suitable agents for each ticket:
        
        Ticket Requirements: {json.dumps(ticket_infos)}
        Available Agents: {json.dumps(agents_json)}
        
        Consider for each ticket:
        1. Skill match (both exact and related skills)
        2. Experience level
        3. Current workload
        
        Return a JSON array where each element is an array of the top {top_k} agents for that ticket.
        Each agent entry should have agent_id and score (0-100).
        Return the response in JSON format only, no additional text."""
        
        response = anthropic.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content[0].text)
    
    def find_best_agents(self, ticket_info, agents, top_k = 3):
        return self.process_batch([ticket_info], agents, top_k)[0]

class AssignmentEngine:
    def __init__(self):
        self.ticket_processor = TicketProcessor()
        self.skill_matcher = SkillMatcher()
    
    def process_batch(self, tickets, agents):
        # Extract ticket information for the batch
        ticket_infos = self.ticket_processor.process_ticket_batch(tickets)
        
        # Find best matching agents for all tickets
        best_agents_batch = self.skill_matcher.process_batch(ticket_infos, agents)
        
        # Create assignments
        assignments = []
        for ticket, best_agents in zip(tickets, best_agents_batch):
            selected_agent = best_agents[0]  # Take the top match
            assignments.append(TicketAssignment(
                ticket_id=ticket.ticket_id,
                title=ticket.title,
                assigned_agent_id=selected_agent["agent_id"],
                rationale=f"Assigned based on skill match ({selected_agent['score']}% match) and current workload."
            ))
        
        return assignments

    def assign_ticket(self, ticket, agents) -> TicketAssignment:
        return self.process_batch([ticket], agents)[0]

def process_batch(engine: AssignmentEngine, batch, agents):
    assignments = engine.process_batch(batch, agents)
    return assignments

def main():
    with open("dataset.json") as f:
        data = json.load(f)
    
    agents = [Agent(**agent_data) for agent_data in data["agents"]]
    tickets = [Ticket(**ticket_data) for ticket_data in data["tickets"]]
    engine = AssignmentEngine()
    assignments = []
    batches = [tickets[i:i + BATCH_SIZE] for i in range(0, len(tickets), BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_batch, engine, batch, agents)
            for batch in batches
        ]
        
        for future in futures:
            batch_assignments = future.result()
            assignments.extend([a.model_dump() for a in batch_assignments])
            
            print(f"Tickets processed: {len(assignments)}")

    output = {"assignments": assignments}
    with open("output_result.json", "w") as f:
        json.dump(output, f, indent=2)    

if __name__ == "__main__":
    main()
