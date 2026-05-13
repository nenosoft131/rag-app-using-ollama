import streamlit as st
import os
import uuid

from api_client import RAGAPIClient

st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("📄 PDF RAG Chat with LangGraph Backend")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── Session state init ─────────────────────────────────────────────────────────
if "api_client" not in st.session_state:
    st.session_state.api_client = RAGAPIClient(API_BASE_URL)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# dict of thread_id -> list of messages (in-memory cache)
if "messages_by_thread" not in st.session_state:
    st.session_state.messages_by_thread = {}

if "chat_threads" not in st.session_state:
    # seed from backend so history survives page refresh
    try:
        sessions = st.session_state.api_client.get_sessions()
        st.session_state.chat_threads = sessions
    except Exception:
        st.session_state.chat_threads = []

# ensure current session is in the list
if st.session_state.session_id not in st.session_state.chat_threads:
    st.session_state.chat_threads.insert(0, st.session_state.session_id)


# ── Helpers ────────────────────────────────────────────────────────────────────
def switch_to_thread(thread_id: str):
    # persist current messages before switching
    st.session_state.messages_by_thread[st.session_state.session_id] = (
        st.session_state.messages
    )
    st.session_state.session_id = thread_id

    # load from cache first, otherwise fetch from backend
    if thread_id in st.session_state.messages_by_thread:
        st.session_state.messages = st.session_state.messages_by_thread[thread_id]
    else:
        try:
            history = st.session_state.api_client.get_chat_history(thread_id)
            st.session_state.messages = history
            st.session_state.messages_by_thread[thread_id] = history
        except Exception:
            st.session_state.messages = []


def new_chat():
    st.session_state.messages_by_thread[st.session_state.session_id] = (
        st.session_state.messages
    )
    new_id = str(uuid.uuid4())
    st.session_state.session_id = new_id
    st.session_state.messages = []
    st.session_state.chat_threads.insert(0, new_id)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    model = st.selectbox(
        "Select Ollama Model", ["llama2", "mistral", "codellama"], index=0
    )

    try:
        doc_info = st.session_state.api_client.get_documents()
        st.success(f"API Connected — {doc_info['document_count']} chunks stored")
    except Exception as e:
        st.error(f"API Connection Error: {str(e)}")
        st.info("Make sure the backend is running on http://localhost:8000")

    if st.button("Clear All Documents"):
        try:
            st.session_state.api_client.clear_documents()
            st.session_state.messages = []
            st.success("All documents cleared")
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")

    if st.button("➕ New Chat"):
        new_chat()
        st.rerun()

    st.header("Chat History")
    for thread_id in st.session_state.chat_threads:
        label = f"Chat {thread_id[:8]}…"
        is_active = thread_id == st.session_state.session_id
        if st.button(
            label,
            key=f"thread_{thread_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            if not is_active:
                switch_to_thread(thread_id)
                st.rerun()

# ── Upload PDF ─────────────────────────────────────────────────────────────────
st.header("Upload PDF")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload a PDF document to use as context for the chat",
)
if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                resp = st.session_state.api_client.upload_document(uploaded_file)
                st.success(resp["message"])
                st.info(f"Total chunks in store: {resp['document_count']}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

# ── Chat interface ─────────────────────────────────────────────────────────────
st.header("Chat with your Document")

for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                for src_idx, source in enumerate(message["sources"]):
                    st.write(f"**Source {src_idx + 1}:**")
                    st.text_area(
                        "", source, height=100,
                        key=f"history_source_{msg_idx}_{src_idx}",
                    )
                    st.divider()

if prompt := st.chat_input("Ask a question about your document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.api_client.chat(
                    message=prompt,
                    session_id=st.session_state.session_id,
                    model=model,
                )
                st.session_state.session_id = response["session_id"]

                assistant_text = response["response"]
                sources = response.get("sources", [])
                st.markdown(assistant_text)

                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text, "sources": sources}
                )
                # keep cache in sync
                st.session_state.messages_by_thread[st.session_state.session_id] = (
                    st.session_state.messages
                )

                if sources:
                    with st.expander("📄 View Sources"):
                        for src_idx, source in enumerate(sources):
                            st.write(f"**Source {src_idx + 1}:**")
                            st.text_area(
                                "", source, height=100,
                                key=f"live_source_{len(st.session_state.messages)}_{src_idx}",
                            )
                            st.divider()

            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
