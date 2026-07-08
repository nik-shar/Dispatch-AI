from graph.state import TicketState
from graph.nodes import *

from langgraph.graph import StateGraph, START, END

graph = StateGraph(TicketState)
graph.add_node("classify", classify_node)
graph.add_node("complaint",complaint_node)
graph.add_node("enquiry",enquiry_node)
graph.add_node("service_request",service_request_node)
graph.add_node("escalation",escalation_node)
graph.add_node("log",log_node)

def route_by_classification(state: TicketState) -> str:
    # Map the classification string to the node name
    class_map = {
        "Complaint": "complaint",
        "Enquiry": "enquiry",
        "Service Request": "service_request",
        "Escalation": "escalation"
    }
    # Fallback to escalation if not matched
    return class_map.get(state.get("classification"), "escalation")

graph.add_edge(START,"classify")

# Conditional routing based on classification
graph.add_conditional_edges(
    "classify",
    route_by_classification,
    {
        "complaint": "complaint",
        "enquiry": "enquiry",
        "service_request": "service_request",
        "escalation": "escalation"
    }
)

graph.add_edge("complaint","log")
graph.add_edge("enquiry","log")
graph.add_edge("service_request","log")
graph.add_edge("escalation","log")

app = graph.compile()

# ==============================================================================
# TEST SCENARIOS
# ==============================================================================

examples = [
    {
        "input_metadata": {"sender": "user1@company.com", "subject": "Portal Outage"},
        "raw_input": (
            "Dear Customer Support Team,\n\nI am writing to report a significant problem with the "
            "centralized account management portal, which currently appears to be offline. "
            "This outage is blocking access to account settings, leading to substantial inconvenience. "
            "I have attempted to log in multiple times using different browsers and devices, "
            "but the issue persists.\n\nCould you please provide an update on the outage status and an "
            "estimated time for resolution?"
        )
    },
    {
        "input_metadata": {"sender": "dev@company.com", "subject": "CI/CD pipeline documentation"},
        "raw_input": (
            "I hope this message finds you well.\n"
            "I am reaching out to request detailed documentation related to the CI/CD pipeline employed "
            "in the current project. Comprehensive information on setup procedures, configurations, "
            "and best practices would be immensely helpful for our development team to optimize workflows "
            "and ensure seamless deployment processes.\n\n"
            "Additionally, if there are sample configuration templates available, please share those as well."
        )
    },
    {
        "input_metadata": {"sender": "newhire@company.com", "subject": "Jira Access Request"},
        "raw_input": (
            "Hi IT team, I recently joined the marketing department and I need access to the Jira workspace "
            "for the Q3 Campaign project. I don't need it today, but I will need to start reviewing tasks "
            "there by Wednesday. Can someone please provision an account for me and assign me to the "
            "Marketing board? Thanks."
        )
    },
    {
        "input_metadata": {"sender": "angry_client@enterprise.com", "subject": "4th Outage this month"},
        "raw_input": (
            "I am writing to formally escalate this issue to your management team IMMEDIATELY. "
            "This is the fourth time our production server has gone down this month, completely halting "
            "our trading floor. We are actively losing thousands of dollars every minute this is down. "
            "The previous support agents assured us this was permanently fixed last week, which was clearly a lie. "
            "I demand a senior executive call me in the next 15 minutes, or we will be pursuing legal action "
            "for breach of SLA."
        )
    }
]

if __name__ == "__main__":
    import pprint
    
    for i, ex in enumerate(examples, 1):
        print(f"\n{'='*80}")
        
        task = TicketState(
            input_source="simulated_inbox",
            raw_input=ex["raw_input"],
            input_metadata=ex["input_metadata"]
        )
        
        result = app.invoke(task)
        
        print(f"\n--> CLASSIFICATION: {result.get('classification')}")
        print(f"--> URGENCY:        {result.get('urgency')}")
        print(f"--> STATUS:         {result.get('status')}")
        print(f"--> ROUTED TO:      {result.get('route_to')}")
        print(f"--> REASONING:      {result.get('reasoning')}")
        
        print("\n--> ACTIONS TAKEN:")
        for action in result.get('actions_taken', []):
            print(f"    {action}")
            
        print("\n--> GENERATED RESPONSE / DRAFT:")
        print(result.get('response_draft', 'None generated'))
        print("\n")