# NeuraNova

A small stateful chatbot built with **LangGraph** and a hosted Hugging Face model.
It keeps conversation memory per session and exposes a FastAPI server.

This is a learning project for graph-based chat state, not a production agent platform.

## Features

- Session-based chat memory with LangGraph `MemorySaver`
- Hugging Face chat model via LangChain
- FastAPI routes for sending messages and fetching a session history

## Tech stack

- Python 3.11+
- LangGraph
- LangChain Hugging Face
- FastAPI + Uvicorn

## Setup

```bash
git clone https://github.com/Naksh7Gupta/NeuraNova.git
cd NeuraNova
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put a Hugging Face token in `.env`:

```bash
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

## Run

```bash
uvicorn server:app --reload
```

## Notes

- Memory is in-process. Restarting the server clears chat history.
- The model ID lives in `main.py` and can be swapped there.
- No RAG and no tools yet. The graph is one chat node.

## License

MIT. See [LICENSE](LICENSE).
