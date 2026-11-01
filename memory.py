"""
Conversation Memory and Context Builder for LocalGPT.
Manages:
1. Short-term active conversation context & window management.
2. Long-term conversation persistence via database.
3. System prompt and RAG context injection into message history.
"""
from typing import List, Dict, Any, Optional
import database


class ConversationMemory:
    def __init__(self, conversation_id: Optional[str] = None):
        if conversation_id:
            self.conversation_id = conversation_id
        else:
            self.conversation_id = database.create_conversation("New Chat")

    def get_id(self) -> str:
        return self.conversation_id

    def switch_conversation(self, new_id: str) -> None:
        self.conversation_id = new_id

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns all messages for the active conversation from the database."""
        return database.get_messages(self.conversation_id)

    def add_user_message(self, content: str) -> str:
        """Adds a user message and auto-generates a title if this is the first message."""
        history = self.get_history()
        msg_id = database.add_message(self.conversation_id, role="user", content=content)
        
        # If this is the first user message, generate title from content
        if len(history) == 0 or (len(history) == 1 and history[0]["role"] == "system"):
            clean_title = content.strip().split("\n")[0][:36]
            if len(content.strip().split("\n")[0]) > 36:
                clean_title += "..."
            database.update_conversation_title(self.conversation_id, clean_title)
            
        return msg_id

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        xray_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Adds an assistant response with optional sources and X-Ray references."""
        return database.add_message(
            self.conversation_id,
            role="assistant",
            content=content,
            sources=sources,
            xray_data=xray_data
        )

    def delete_from_message(self, message_id: str, include_self: bool = True) -> None:
        """Deletes messages starting from message_id onwards."""
        database.delete_messages_from(self.conversation_id, message_id, include_self=include_self)

    def build_prompt_context(
        self,
        system_prompt: str = "You are a helpful AI assistant. Explain concepts clearly with examples.",
        rag_context: str = "",
        max_history_turns: int = 10
    ) -> List[Dict[str, str]]:
        """
        Builds the structured message list for HuggingFace chat template:
        [
            {"role": "system", "content": ...},
            {"role": "user", "content": ...},
            {"role": "assistant", "content": ...},
            ...
        ]
        Injects system prompt and RAG retrieved context appropriately.
        """
        messages: List[Dict[str, str]] = []
        
        # 1. System Prompt + RAG Context
        full_system = system_prompt.strip()
        if rag_context.strip():
            full_system += (
                "\n\nYou have access to the following reference documents. "
                "Base your answers on this context when relevant, and cite document sources:\n\n"
                f"{rag_context.strip()}"
            )
        
        messages.append({"role": "system", "content": full_system})

        # 2. Fetch history and trim to window
        history = self.get_history()
        
        # Filter out system messages from history to avoid duplicates
        chat_turns = [m for m in history if m["role"] in ("user", "assistant")]
        
        # Keep last max_history_turns turns
        if max_history_turns and len(chat_turns) > max_history_turns * 2:
            chat_turns = chat_turns[-(max_history_turns * 2):]

        for m in chat_turns:
            messages.append({
                "role": m["role"],
                "content": m["content"]
            })

        return messages

