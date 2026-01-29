# LangGraph Chatbot with HuggingFace LLM

A simple AI chatbot built using **LangGraph** and **HuggingFace LLMs**, with memory persistence for conversation state. Perfect for building modular and stateful AI assistants.

---

## Features

- 🧠 **Stateful Chat:** Tracks conversation state using LangGraph.
- 🤖 **LLM-powered Responses:** Uses HuggingFace `openai/gpt-oss-20b` model for AI replies.
- 💾 **Memory Persistence:** Stores conversation history with `MemorySaver`.
- ⚡ **Modular & Clean:** Easy to replace LLM or add new nodes to the chat graph.

---

## Tech Stack

- **Python 3.11+**
- **LangGraph** – Manage chat states and flows
- **LangChain-HuggingFace** – LLM integration
- **HuggingFace Endpoint** – Provider for AI responses
- **dotenv** – Environment variable management

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/langgraph-chatbot.git
cd langgraph-chatbot
