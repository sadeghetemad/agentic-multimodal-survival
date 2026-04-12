from agent.tools import TOOLS
import json


def execute_tool(plan, state):

    
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return {"status": "error", "message": "Invalid plan format"}

    if isinstance(plan, dict):
        plan = [plan]

    if not isinstance(plan, list):
        return {"status": "error", "message": "Plan must be a list"}

    last_result = None

    for step in plan:

        if isinstance(step, str):
            try:
                step = json.loads(step)
            except Exception:
                return {"status": "error", "message": "Invalid step format"}

        if not isinstance(step, dict):
            return {"status": "error", "message": "Step must be a dict"}

        action = step.get("action")
        tool_input = step.get("input", {})

        if isinstance(tool_input, str):
            tool_input = {"text": tool_input}
        elif not isinstance(tool_input, dict):
            tool_input = {}

        if "user_input" in state and "text" not in tool_input:
            tool_input["text"] = state.get("user_input", "")

        if "features" in state and state["features"] and "features" not in tool_input:
            tool_input["features"] = state["features"]

        if action not in TOOLS:
            return {"status": "error", "message": f"Unknown tool: {action}"}

        tool = TOOLS[action]

        try:
            result = tool(tool_input)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        if not isinstance(result, dict):
            return {"status": "error", "message": "Invalid tool output"}

        if result.get("status") == "error":
            return result

        if "features" in result:
            state["features"] = result["features"]

        last_result = result

    return last_result