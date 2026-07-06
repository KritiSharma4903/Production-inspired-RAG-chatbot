import os
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Production RAG Chatbot", layout="wide")
st.title("📄 Production RAG Chatbot")

tab_upload, tab_chat = st.tabs(["Upload / Update Document", "Chat"])

with tab_upload:
    st.subheader("Upload a PDF or DOCX")
    st.write("Uploading the same filename again will update the document.")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx"]
    )

    if uploaded_file and st.button("Ingest"):
        extension = os.path.splitext(uploaded_file.name)[1].lower()
        if extension == ".pdf":
            content_type = "application/pdf"
        elif extension == ".docx":
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            st.error("Unsupported file type.")
            st.stop()
        with st.spinner("Uploading and processing..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    content_type
                )
            }

            response = requests.post(
                f"{API_BASE}/documents/upload",
                files=files
            )

        if response.ok:
            data = response.json()
            st.success("Document processed successfully.")
            st.json(data)

        else:
            st.error(response.text)


with tab_chat:
    st.subheader("Ask Questions")
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    question = st.chat_input("Ask something...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):

            response = requests.post(
                f"{API_BASE}/chat/ask",
                json={
                    "question": question,
                    "session_id": st.session_state.session_id
                }
            )

        if response.ok:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            answer = data["answer"]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )
            with st.chat_message("assistant"):
                st.write(answer)
                if data.get("sources"):
                    with st.expander("Sources"):
                        for source in data["sources"]:
                            st.write(
                                f"Page {source['page']} | Score: {source['score']:.3f}"
                            )
        else:
            st.error(response.text)