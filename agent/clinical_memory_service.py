from datetime import datetime, UTC
from typing import Dict, Any, List
import boto3

from config.settings import (
    AWS_REGION,
    MEMORY_ENABLED,
    MEMORY_ID,
    MEMORY_SUMMARY_TEMPLATE,
    MEMORY_PREFERENCES_TEMPLATE,
)


class ClinicalMemoryService:

    def __init__(self):
        self.enabled_flag = MEMORY_ENABLED and bool(MEMORY_ID)
        self.memory_id = MEMORY_ID

        self.summary_template = MEMORY_SUMMARY_TEMPLATE
        self.preferences_template = MEMORY_PREFERENCES_TEMPLATE

        self.client = boto3.client(
            "bedrock-agentcore",
            region_name=AWS_REGION
        )

    def enabled(self) -> bool:
        return self.enabled_flag

    def render_summary_namespace(self, actor_id: str, session_id: str) -> str:
        return self.summary_template.format(
            actor_id=actor_id,
            session_id=session_id
        )

    def render_preferences_namespace(self, actor_id: str) -> str:
        return self.preferences_template.format(
            actor_id=actor_id
        )

    # WRITE CONVERSATION EVENT
    def capture_turn(
        self,
        actor_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str
    ) -> None:
        """
        Send raw conversational turns to AgentCore Memory.
        Built-in strategies will extract long-term memories asynchronously.
        """
        if not self.enabled():
            return

        payload = []

        if user_text and user_text.strip():
            payload.append({
                "conversational": {
                    "role": "USER",
                    "content": {
                        "text": user_text.strip()
                    }
                }
            })

        if assistant_text and assistant_text.strip():
            payload.append({
                "conversational": {
                    "role": "ASSISTANT",
                    "content": {
                        "text": assistant_text.strip()
                    }
                }
            })

        if not payload:
            return

        try:
            self.client.create_event(
                memoryId=self.memory_id,
                actorId=actor_id,
                sessionId=session_id,
                eventTimestamp=datetime.now(UTC),
                payload=payload
            )
        except Exception as e:
            print(f"[ClinicalMemoryService] create_event error: {e}")


    # LONG-TERM: PREFERENCES
    def get_preferences(
        self,
        actor_id: str,
        query: str = "clinician output preferences"
    ) -> Dict[str, Any]:
        """
        Retrieve actor-level preferences extracted by your built-in strategy.
        """
        if not self.enabled():
            return {}

        namespace = self.render_preferences_namespace(actor_id)

        try:
            response = self.client.retrieve_memory_records(
                memoryId=self.memory_id,
                namespace=namespace,
                searchCriteria={
                    "searchQuery": query,
                    "topK": 5
                }
            )

            records = response.get("memoryRecordSummaries", [])
            texts = []

            for item in records:
                content = item.get("content", {})
                text = content.get("text")
                if text:
                    texts.append(text)

            return {
                "raw_texts": texts,
                "parsed": self._parse_preferences_texts(texts)
            }

        except Exception as e:
            print(f"[ClinicalMemoryService] get_preferences error: {e}")
            return {}


    # LONG-TERM: SESSION SUMMARY
    def get_session_summary(
        self,
        actor_id: str,
        session_id: str,
        query: str = "summary of this session"
    ) -> Dict[str, Any]:
        """
        Retrieve session-scoped summaries extracted by your summary strategy.
        """
        if not self.enabled():
            return {}

        namespace = self.render_summary_namespace(actor_id, session_id)

        try:
            response = self.client.retrieve_memory_records(
                memoryId=self.memory_id,
                namespace=namespace,
                searchCriteria={
                    "searchQuery": query,
                    "topK": 5
                }
            )

            records = response.get("memoryRecordSummaries", [])
            texts = []

            for item in records:
                content = item.get("content", {})
                text = content.get("text")
                if text:
                    texts.append(text)

            return {
                "raw_texts": texts
            }

        except Exception as e:
            print(f"[ClinicalMemoryService] get_session_summary error: {e}")
            return {}


    # SIMPLE HEURISTIC PARSER
    def _parse_preferences_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        Lightweight parser for preference memories returned as text.
        You can improve this later with an LLM parser if needed.
        """
        merged: Dict[str, Any] = {}

        joined = " ".join(texts).lower()

        if "concise" in joined or "brief" in joined:
            merged["response_style"] = "concise"

        if "detailed" in joined or "detail" in joined:
            merged["response_style"] = "detailed"

        if "technical" in joined or "clinical terminology" in joined:
            merged["tone"] = "technical"

        if "plain" in joined or "simple language" in joined:
            merged["tone"] = "plain"

        if "top 3" in joined:
            merged["top_k_features"] = 3
        elif "top 5" in joined:
            merged["top_k_features"] = 5

        if "do not show top features" in joined or "hide top features" in joined:
            merged["show_top_features"] = False
        elif "show top features" in joined or "include top features" in joined:
            merged["show_top_features"] = True

        return merged