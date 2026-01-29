from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
)
# from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# -------------------- LLM --------------------
model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation', provider='together')
)

# -------------------- STATE --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------- HELPERS --------------------
def summarize_messages(messages: list[BaseMessage]) -> str:
    """
    Summarize last 4 messages into a compact memory
    """
    text = ""
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text += f"{role}: {m.content}\n"

    prompt = f"""
    Summarize the following conversation briefly
    so it can be used as memory later.

    Conversation:
    {text}
    """

    summary = model.invoke(
        [HumanMessage(content=prompt)]
    )

    return summary.content


def apply_stm_strategy(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    If messages > 10:
    - summarize last 4 messages
    - keep summary + last 5 messages
    """
    if len(messages) <= 10:
        return messages

    # Last 4 messages for summary
    to_summarize = messages[-4:]
    summary_text = summarize_messages(to_summarize) 

    summary_message = SystemMessage(
        content=f"Conversation summary so far:\n{summary_text}"
    )

    # Keep last 5 messages
    recent_messages = messages[-5:]

    return [summary_message, *recent_messages]

# -------------------- CHAT NODE --------------------
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]

    # 🧠 Apply STM logic
    messages_for_llm = apply_stm_strategy(messages)

    response = model.invoke(
        [
            SystemMessage(content="You are a helpful AI assistant."),
            *messages_for_llm
        ]
    )

    return {"messages": [response]}

# -------------------- GRAPH --------------------
graph = StateGraph(ChatState)

graph.add_node("Chat", chat_node)
graph.add_edge(START, "Chat")

# -------------------- MEMORY --------------------
checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)
