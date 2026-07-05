"""
RAGEngine protocol.

Why this exists: ChatService and QAService need to call "ask a question" and
"get predefined questions" — but the current implementation of those two
things lives inside ai_assistant.py's AIAssistant god-object (690 lines,
singleton, handles embeddings + LLM loading + prompt building + retrieval in
one class). That class is refactored in a later Phase 1 increment
(app/infrastructure/llm/).

Rather than have services import AIAssistant directly (which would just move
the coupling problem one layer over), services depend on this narrow
Protocol instead. AIAssistant already happens to satisfy it structurally
(duck typing — Python Protocols don't require inheritance), so it can be
passed in as-is via dependency injection today, and swapped for the
refactored engine later with no change to ChatService/QAService at all.
"""
from typing import Any, Dict, List, Protocol


class RAGEngine(Protocol):
    async def process_query(
        self, query: str, tenant_id: str, customer_id: str, lang: str, session_id: str
    ) -> Dict[str, Any]:
        """Returns {"answer": str, "sources": list}."""
        ...

    async def get_questions(self, tenant_id: str, lang: str) -> List[Dict[str, Any]]:
        """Returns a list of {"question": str, "created_at": datetime | None}."""
        ...
