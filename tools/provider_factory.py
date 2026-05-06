from config.settings import TOOL_PROVIDER
from tools.local_tool_provider import LocalToolProvider
from tools.gateway_tool_provider import GatewayToolProvider


_provider = None


def get_tool_provider():
    global _provider

    if _provider is not None:
        return _provider

    if TOOL_PROVIDER == "gateway":
        print("[ToolProvider] Using GatewayToolProvider")
        _provider = GatewayToolProvider()
    else:
        print("[ToolProvider] Using LocalToolProvider")
        _provider = LocalToolProvider()

    return _provider