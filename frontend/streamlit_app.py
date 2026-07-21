import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Production RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🤖 Production RAG")

    st.markdown("---")

    st.subheader("Backend")

    try:

        response = requests.get(f"{API_BASE}/health")

        if response.status_code == 200:
            st.success("Backend Running")

        else:
            st.error("Backend Offline")

    except Exception:
        st.error("Cannot connect")

    st.markdown("---")

    st.subheader("Session")

    if st.session_state.session_id:

        st.code(st.session_state.session_id)

    else:

        st.info("No active session")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.last_answer = ""
        st.session_state.last_question = ""
        st.session_state.last_contexts = []
        st.session_state.last_sources = []

        st.rerun()

# -----------------------------
# Main Title
# -----------------------------

st.title("🤖 Production RAG Chatbot")

st.caption("Upload → Ask → Evaluate")

# ==========================================================
# Upload Document Section
# ==========================================================

st.markdown("---")
st.header("📄 Upload / Update Document")

upload_col1, upload_col2 = st.columns([3, 1])

with upload_col1:

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Uploading the same filename again will update the document."
    )

with upload_col2:

    st.write("")
    st.write("")
    upload_btn = st.button(
        "🚀 Upload",
        use_container_width=True,
        type="primary"
    )


if upload_btn:

    if uploaded_file is None:

        st.warning("Please select a PDF file first.")

    else:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }

        progress = st.progress(0)

        status = st.empty()

        try:

            status.info("Uploading document...")

            progress.progress(20)

            response = requests.post(
                f"{API_BASE}/documents/upload",
                files=files,
                timeout=300,
            )

            progress.progress(80)

            if response.status_code == 200:

                result = response.json()

                progress.progress(100)

                status.success("Document processed successfully.")

                st.success(
                    f"✅ {uploaded_file.name} uploaded successfully."
                )

                st.markdown("### Document Information")

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Status",
                        result.get("status", "-"),
                    )

                with c2:

                    st.metric(
                        "Version",
                        result.get("version", "-"),
                    )

                with c3:

                    st.metric(
                        "Embedded Chunks",
                        result.get("chunks_embedded", "-"),
                    )

                with st.expander("Complete Response"):

                    st.json(result)

            else:

                progress.empty()

                status.error(response.text)

        except requests.exceptions.ConnectionError:

            progress.empty()

            st.error("Cannot connect to FastAPI backend.")

        except Exception as e:

            progress.empty()

            st.exception(e)

st.markdown("---")

# ==========================================================
# Chat Section
# ==========================================================

st.header("💬 Chat with your Documents")

# Display previous chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat Input
question = st.chat_input("Ask a question about your document...")


if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Loading animation
    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.info("Searching relevant chunks...")

        try:

            payload = {
                "question": question,
                "session_id": st.session_state.session_id,
            }

            response = requests.post(
                f"{API_BASE}/chat/ask",
                json=payload,
                timeout=300,
            )

            if response.status_code != 200:

                thinking.error(response.text)

            else:

                data = response.json()

                answer = data.get("answer", "")

                contexts = data.get("contexts", [])

                sources = data.get("sources", [])

                session_id = data.get("session_id")

                # Save session
                st.session_state.session_id = session_id

                # Save latest answer for evaluation
                st.session_state.last_question = question

                st.session_state.last_answer = answer

                st.session_state.last_contexts = contexts

                st.session_state.last_sources = sources

                # Remove loading
                thinking.empty()

                # Show answer
                st.markdown(answer)

                # Save assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # ------------------------------------------------
                # Retrieved Sources
                # ------------------------------------------------

                if sources:

                    with st.expander("📚 Retrieved Sources", expanded=False):

                        for i, src in enumerate(sources, start=1):

                            st.markdown(
                                f"""
**Chunk {i}**

- Chunk ID : `{src.get("chunk_id")}`

- Page : {src.get("page")}

- Similarity Score : {round(src.get("score",0),4)}
"""
                            )

                # ------------------------------------------------
                # Retrieved Contexts
                # ------------------------------------------------

                if contexts:

                    with st.expander("📄 Retrieved Context", expanded=False):

                        for i, ctx in enumerate(contexts, start=1):

                            st.markdown(f"### Context {i}")

                            st.write(ctx)

                            st.divider()

        except Exception as e:

            thinking.empty()

            st.error(str(e))


# ==========================================================
# RAG Evaluation
# ==========================================================

st.markdown("---")
st.header("📊 Evaluate Last Answer")

if not st.session_state.last_answer:

    st.info("Ask at least one question before running evaluation.")

else:

    st.success("Latest answer is ready for evaluation.")

    st.markdown("### Latest Question")

    st.info(st.session_state.last_question)

    st.markdown("### Generated Answer")

    st.write(st.session_state.last_answer)

    ground_truth = st.text_area(
        "Ground Truth (Optional)",
        height=150,
        placeholder="Paste the expected / ideal answer here..."
    )

    evaluate_btn = st.button(
        "🚀 Evaluate Answer",
        type="primary",
        use_container_width=True,
    )

    if evaluate_btn:

        payload = {

            "question": st.session_state.last_question,

            "answer": st.session_state.last_answer,

            "contexts": st.session_state.last_contexts,

            "ground_truth": ground_truth if ground_truth else None,
        }

        with st.spinner("Running Evaluation..."):

            try:

                response = requests.post(
                    f"{API_BASE}/evaluation/single",
                    json=payload,
                    timeout=300,
                )

                if response.status_code != 200:

                    st.error(response.text)

                else:

                    result = response.json()

                    st.success("Evaluation Completed")

                    st.markdown("---")

                    st.subheader("📈 Evaluation Metrics")

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Context Precision",
                            f"{(result.get('context_precision') or 0):.3f}",
                        )

                    with col2:

                        st.metric(
                            "Context Recall",
                            f"{(result.get('context_recall') or 0):.3f}",
                        )

                    with col3:

                        st.metric(
                            "Faithfulness",
                            f"{(result.get('faithfulness') or 0):.3f}",
                        )

                    col4, col5, col6 = st.columns(3)

                    with col4:

                        st.metric(
                            "Answer Relevancy",
                            f"{(result.get('answer_relevancy') or 0):.3f}",
                        )

                    with col5:

                        st.metric(
                            "Correctness",
                            f"{(result.get('correctness') or 0):.3f}",
                        )

                    with col6:

                        st.metric(
                            "Latency (sec)",
                            f"{(result.get('latency') or 0):.2f}",
                        )

                    st.markdown("---")

                    st.subheader("📊 Metric Scores")

                    cp = result.get("context_precision") or 0
                    cr = result.get("context_recall") or 0
                    faith = result.get("faithfulness") or 0
                    rel = result.get("answer_relevancy") or 0
                    corr = result.get("correctness") or 0

                    st.write("Context Precision")
                    st.progress(float(cp))

                    st.write("Context Recall")
                    st.progress(float(cr))

                    st.write("Faithfulness")
                    st.progress(float(faith))

                    st.write("Answer Relevancy")
                    st.progress(float(rel))

                    st.write("Correctness")
                    st.progress(float(corr))

                    st.markdown("---")

                    overall = (
                        cp +
                        cr +
                        faith +
                        rel +
                        corr
                    ) / 5

                    st.subheader("🏆 Overall RAG Score")

                    st.metric(
                        "Overall Score",
                        f"{overall:.3f}",
                    )

                    if overall >= 0.90:

                        st.success("Excellent RAG Pipeline ✅")

                    elif overall >= 0.75:

                        st.info("Good Performance 👍")

                    elif overall >= 0.50:

                        st.warning("Needs Improvement ⚠️")

                    else:

                        st.error("Poor Retrieval / Generation ❌")

                    with st.expander("Raw Evaluation JSON"):

                        st.json(result)

            except Exception as e:

                st.exception(e)


# ==========================================================
# Footer Dashboard
# ==========================================================

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Questions Asked",
        len(
            [
                m
                for m in st.session_state.messages
                if m["role"] == "user"
            ]
        ),
    )

with col2:

    st.metric(
        "Answers Generated",
        len(
            [
                m
                for m in st.session_state.messages
                if m["role"] == "assistant"
            ]
        ),
    )

with col3:

    st.metric(
        "Retrieved Contexts",
        len(st.session_state.last_contexts),
    )

with col4:

    if st.session_state.session_id:

        st.success("Active Session")

    else:

        st.warning("No Session")

st.markdown("---")

with st.expander("📋 Session Information"):

    st.write("### Session ID")

    st.code(
        st.session_state.session_id
        if st.session_state.session_id
        else "No Session"
    )

    st.write("### Last Question")

    st.write(
        st.session_state.last_question
        if st.session_state.last_question
        else "-"
    )

    st.write("### Last Answer")

    st.write(
        st.session_state.last_answer
        if st.session_state.last_answer
        else "-"
    )

    st.write("### Retrieved Chunks")

    st.write(len(st.session_state.last_contexts))

st.markdown("---")

with st.expander("💾 Download Conversation"):

    import json

    conversation = {

        "session_id": st.session_state.session_id,

        "messages": st.session_state.messages,

        "last_question": st.session_state.last_question,

        "last_answer": st.session_state.last_answer,

        "contexts": st.session_state.last_contexts,

        "sources": st.session_state.last_sources,

    }

    st.download_button(

        "Download Chat JSON",

        json.dumps(conversation, indent=4),

        file_name="conversation.json",

        mime="application/json",

    )

st.markdown("---")

st.caption(
    "🚀 Production RAG Chatbot | FastAPI • Pinecone • Groq • PostgreSQL • Streamlit"
)



