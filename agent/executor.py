import json
from agent.tools import TOOLS

def execute_tool(plan):
    
    action = plan["action"]
    tool_input = plan["input"]

    tool = TOOLS[action]

    result = tool.invoke(json.dumps(tool_input))

    return result