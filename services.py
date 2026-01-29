from langchain_core.messages import HumanMessage
from main import chatbot
from schemas import GetChat


def run_chatbot(session_id: str, user_message: str) -> str:
    """
    Runs chatbot with ONLY session-based memory (NO RAG)
    """

    # 🔑 LangGraph thread-based memory
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    result = chatbot.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )

    return result["messages"][-1].content


# ================= FETCH CHAT BY SESSION =================
def get_chats(request: GetChat):
    session_id = request.session_id

    # 🔥 get state using session/thread id
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": session_id
            }
        }
    )

    messages = state.values.get("messages", []) if state else []

    # Convert messages to clean JSON
    clean_messages = [
        {
            "role": msg.type,
            "content": msg.content
        }
        for msg in messages
    ]

    return {
        "session_id": session_id,
        "messages": clean_messages
    }
