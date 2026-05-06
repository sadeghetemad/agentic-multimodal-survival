from functools import lru_cache
from typing import Optional

from langgraph_checkpoint_aws import AgentCoreMemorySaver
from config.settings import MEMORY_ID, AWS_REGION, MEMORY_ENABLED


@lru_cache(maxsize=1)
def get_checkpointer() -> Optional[AgentCoreMemorySaver]:
    if not MEMORY_ENABLED or not MEMORY_ID:
        return None

    return AgentCoreMemorySaver(
        memory_id=MEMORY_ID,
        region_name=AWS_REGION
    )