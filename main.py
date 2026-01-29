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
from dotenv import load_dotenv

load_dotenv()

# -------------------- LLM --------------------
model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', task='text-generation', provider='together')
)

# -------------------- STATE --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------- CHAT NODE --------------------
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]

    # Directly use all messages, no STM logic
    response = model.invoke(
        [
            SystemMessage(content="You are a helpful AI assistant."),
            *messages
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
