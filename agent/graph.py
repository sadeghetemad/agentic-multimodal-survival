from langgraph.graph import StateGraph

from agent.planner import plan
from agent.executor import execute_tool

# state
class AgentState(dict):
    pass

# nodes
def planner_node(state):
    plan_output = plan(state["user_input"], state["features"])
    return {"plan": plan_output}

def tool_node(state):
    result = execute_tool(state["plan"])
    return {"tool_result": result}

def response_node(state):
    return {
        "response": f"Result: {state['tool_result']}"
    }

# graph
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "tool")
    graph.add_edge("tool", "response")

    return graph.compile()