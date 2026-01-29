import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

from main import chatbot
from schemas import GetChat

UPLOAD_FOLDER = "uploads"
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def run_chatbot(session_id: str, user_message: str) -> str:
    """
    Runs chatbot with RAG (if PDF exists) + session-based memory
    """

    # 🔑 CONFIG (LangGraph / thread-based memory)
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    session_pdf_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.pdf")

    # ================= RAG FLOW =================
    if os.path.exists(session_pdf_path):

        # 1️⃣ Load PDF
        loader = PyPDFLoader(session_pdf_path)
        docs = loader.load()

        # 2️⃣ Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # 3️⃣ Vector Store + Retriever
        vectorstore = FAISS.from_documents(chunks, embed_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 4️⃣ Retrieve relevant context
        relevant_docs = retriever.invoke(user_message)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)    

        human_message = f"""
        Answer the question from the provided context below.
        
        CONTEXT:
        {context}

        QUESTION:
        {user_message}
        """

        messages = [
            HumanMessage(content=human_message)
        ]

        result = chatbot.invoke(
            {"messages": messages},
            config=config
        )

        return result["messages"][-1].content

    # ================= NORMAL CHAT =================
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

    # 🧠 ONLY return messages (NO metadata)
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
