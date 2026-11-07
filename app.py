import streamlit as st
from pubmed import get_medical_evidence
from llm_s import healthcare_llama, classify_question
from pdf_s import extract_text_from_pdf
from preprocess import (
    extract_text_from_image,
    chunk_text,
    select_relevant_chunks
)
from PIL import Image

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="Healthcare AI Assistant", layout="wide")
st.title("Healthcare Evidence-Based Assistant")
st.warning("This system provides evidence-based information. Not medical advice.")

# ------------------ SESSION STATE INITIALIZATION ------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "chat_title" not in st.session_state:
    st.session_state.chat_title = "Chat 1"

# ---------------- DISPLAY PREVIOUS MESSAGES ---------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ SIDEBAR ------------------ #
st.sidebar.title(" Chat Bar")

# New Chat
if st.sidebar.button("New Chat"):
    if st.session_state.messages:
        st.session_state.history.append({
            "title": st.session_state.chat_title,
            "messages": st.session_state.messages.copy()
        })
    st.session_state.messages = []
    st.session_state.chat_title = f"Chat {len(st.session_state.history) + 1}"
    st.rerun()

# Load Previous Chats
if st.session_state.history:
    st.sidebar.markdown("**Previous Chats:**")

    chat_titles = [chat["title"] for chat in st.session_state.history]

    selected_chat = st.sidebar.radio(
        "Select chat:",
        options=["Current Chat"] + chat_titles
    )

    if selected_chat != "Current Chat":
        for chat in st.session_state.history:
            if chat["title"] == selected_chat:
                st.session_state.messages = chat["messages"].copy()
                st.session_state.chat_title = chat["title"]
                st.rerun()

# Clear All
if st.sidebar.button(" Clear All Chats"):
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.chat_title = "Chat 1"
    st.rerun()

st.sidebar.divider()

# ------------------ INPUT SECTION ------------------ #
st.subheader("Ask a medical question or upload a file")

uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"]
)

query = st.chat_input("Type a medical question...")

# ------------------ HANDLE INPUT ------------------ #
if query or uploaded_file:

    # -------- Determine Input Source -------- #
    if uploaded_file is not None:
        file_type = uploaded_file.type

        if "pdf" in file_type:
            with st.spinner("Extracting text from PDF..."):
                query_text = extract_text_from_pdf(uploaded_file)

        elif "image" in file_type:
            with st.spinner("Processing image & extracting text..."):
                image = Image.open(uploaded_file)
                query_text = extract_text_from_image(image)

        else:
            query_text = ""

        if not query_text:
            st.error(" Could not extract text from uploaded file.")
            st.stop()

    else:
        query_text = query

    # ---------------- SMART CHUNKING ---------------- #
    if len(query_text) > 1500:
        chunks = chunk_text(query_text)
        relevant_chunks = select_relevant_chunks(query_text, chunks)
        query_text = " ".join(relevant_chunks)

    # ---------------- SAVE & DISPLAY USER MESSAGE ---------------- #
    st.session_state.messages.append({
        "role": "user",
        "content": query_text
    })

    with st.chat_message("user"):
        st.markdown(query_text)

    # ---------------- PROCESS QUERY ---------------- #
    with st.chat_message("assistant"):
        with st.spinner("Retrieving medical evidence..."):

            # Classify Question
            q_type = classify_question(query_text)

            abstracts, pmids = get_medical_evidence(query_text)


            # Generate Answer
            if not abstracts:
                answer = "No relevant medical evidence was found."
            else:
                confidence = (
                    "High" if len(pmids) >= 2 else
                    "Medium" if len(pmids) == 1 else
                    "Low"
                )

                answer = healthcare_llama(query_text, abstracts, pmids)

                answer += f"\n\n**Confidence Level:** {confidence}"

                if pmids:
                    answer += "\n\n**Supporting PubMed Articles:**\n"
                    for pmid in pmids:
                        answer += f"- https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"

        st.markdown(answer)

    # ---------------- SAVE ASSISTANT MESSAGE ---------------- #
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
