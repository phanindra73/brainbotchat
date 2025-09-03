import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Setup
# ----------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")  # 🔑 Ensure you set this in your environment
if not openai_api_key:
    st.error("⚠️ Please set the environment variable: open_apikey")
    st.stop()

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

persist_directory = "chroma_store"

# ----------------------------
# Function to load and add PDF to Chroma
# ----------------------------
def add_pdf_to_chroma(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    return vectordb

# ----------------------------
# Load existing or empty Chroma
# ----------------------------
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🤖 BrainBots RAG Chatbot", page_icon="🤖")
st.title("🤖 BrainBots RAG Chatbot")
st.caption("Ask questions from your documents (BrainBots Policy PDF or others)")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    with open("uploaded_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    vectordb = add_pdf_to_chroma("uploaded_doc.pdf")
    st.success("✅ PDF added to knowledge base!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about BrainBots policies..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    docs = vectordb.similarity_search(prompt, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Generate answer
    final_prompt = f"Answer the question based on context:\n\n{context}\n\nQuestion: {prompt}"
    response = llm.invoke(final_prompt).content

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
