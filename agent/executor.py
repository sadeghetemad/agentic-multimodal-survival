from agent.tools import TOOLS

def execute_tool(plan):

    action = plan["action"]
    tool_input = plan["input"]

    if action not in TOOLS:
        return {
            "status": "error",
            "message": f"Unknown tool: {action}"
        }

    tool = TOOLS[action]

    try:
        result = tool(tool_input)
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    return result