import streamlit as st
import os

from api_client import RAGAPIClient

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Chat",
    layout="wide"
)

st.title("📄 PDF RAG Chat with LangGraph Backend")

# --------------------------------------------------
# Configuration
# --------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "api_client" not in st.session_state:
    st.session_state.api_client = RAGAPIClient(API_BASE_URL)

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    model = st.selectbox(
        "Select Ollama Model",
        ["llama2", "mistral", "codellama"],
        index=0
    )

    try:
        response = st.session_state.api_client.get_documents()
        st.success(f"API Connected - {response['document_count']} documents stored")
    except Exception as e:
        st.error(f"API Connection Error: {str(e)}")
        st.info("Make sure the backend is running on http://localhost:8000")

    if st.button("Clear All Documents"):
        try:
            st.session_state.api_client.clear_documents()
            st.session_state.messages = []
            st.session_state.session_id = None
            st.success("All documents cleared")
        except Exception as e:
            st.error(f"Error clearing documents: {str(e)}")

# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
st.header("Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload a PDF document to use as context for the chat"
)

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                response = st.session_state.api_client.upload_document(uploaded_file)
                st.success(response["message"])
                st.info(f"Total documents in store: {response['document_count']}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

# --------------------------------------------------
# Chat interface
# --------------------------------------------------
st.header("Chat with your Document")

# --------------------------------------------------
# Display chat history (STABLE KEYS)
# --------------------------------------------------
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                for src_idx, source in enumerate(message["sources"]):
                    st.write(f"**Source {src_idx + 1}:**")
                    st.text_area(
                        "",
                        source,
                        height=100,
                        key=f"history_source_{msg_idx}_{src_idx}"
                    )
                    st.divider()

# --------------------------------------------------
# Chat input
# --------------------------------------------------
if prompt := st.chat_input("Ask a question about your document"):
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.api_client.chat(
                    message=prompt,
                    session_id=st.session_state.session_id,
                    model=model
                )

                st.session_state.session_id = response["session_id"]

                assistant_text = response["response"]
                sources = response.get("sources", [])

                st.markdown(assistant_text)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                    "sources": sources
                })

                # Display sources immediately (different key namespace)
                if sources:
                    with st.expander("📄 View Sources"):
                        for src_idx, source in enumerate(sources):
                            st.write(f"**Source {src_idx + 1}:**")
                            st.text_area(
                                "",
                                source,
                                height=100,
                                key=f"live_source_{len(st.session_state.messages)}_{src_idx}"
                            )
                            st.divider()

            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })

