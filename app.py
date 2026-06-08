# app.py - User Interface

import streamlit as st
from src.pipeline import RAGPipeline
import os

# Page config
st.set_page_config(
    page_title="Ask My Docs 📚",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Ask My Docs")
st.subheader("Ask Questions related to documents!")


# Pipeline-ai cache pannrom - ovvoru question-kum reload aagaama
@st.cache_resource
def get_pipeline():
    """
    # Pipeline oru time mattum initialize aagum
    # @st.cache_resource idha ensure pannum
    """
    pipeline = RAGPipeline(
        data_folder="./data",
        vector_db_path="./chroma_db"
    )
    pipeline.setup()
    return pipeline


# Sidebar - Document upload
with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "PDF files upload pannu",
        type=["pdf", "txt"],
        accept_multiple_files=True  # Multiple files OK
    )

    if uploaded_files:
        # Uploaded files-ai data folder-la save pannu
        os.makedirs("./data", exist_ok=True)
        for file in uploaded_files:
            with open(f"./data/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)} files uploaded!")

    # Re-index button
    if st.button("🔄 Re-index Documents"):
        st.cache_resource.clear()  # Cache clear pannu
        st.rerun()

# Main chat area
st.header("💬 Ask a Question")

# Question input
question = st.text_input(
    "Unnoda question type pannu:",
    placeholder="Example: What is the leave policy?"
)

if st.button("🔍 Ask", type="primary") and question:

    with st.spinner("Searching... ⏳"):
        try:
            # Pipeline get pannu
            pipeline = get_pipeline()

            # Answer get pannu
            result = pipeline.query(question)

            # Answer display pannu
            st.success("✅ Answer Found!")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 💡 Answer")
                st.markdown(result["answer"])

            with col2:
                st.markdown("### 📖 Sources Used")
                st.info(f"**{result['num_sources']} documents** used")

                for num, cite in result["citations"].items():
                    st.markdown(f"📄 **Doc {num}:** {cite['filename']} (Page {cite['page']})")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Documents upload pannirukiya? Sidebar-la check pannu!")