import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Cyber Security Analyst", page_icon="🛡️", layout="centered")

# --- UI Styling (Clean Professional White) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .main-title { color: #1e3a8a; font-size: 40px; font-weight: bold; text-align: center; }
    .sub-title { color: #4b5563; text-align: center; margin-bottom: 40px; font-size: 18px; font-style: italic; }
    .stChatMessage { background-color: #f3f4f6 !important; border-radius: 12px; border: 1px solid #e5e7eb; color: #111827 !important; }
    input { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛡️ Cyber Security Analyst</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Specialized in NIST Frameworks, Zero Trust & Threat Intelligence</div>", unsafe_allow_html=True)

# --- Database Loading with Auto-Ingestion Logic ---
@st.cache_resource
def load_cyber_db():
    DB_PATH = "./chroma_db_cyber"
    DATA_PATH = "cyber_data/"
    
    # AGAR DATABASE NAHI HAI, TOH KHUD BANAO
    if not os.path.exists(DB_PATH):
        if os.path.exists(DATA_PATH) and len(os.listdir(DATA_PATH)) > 0:
            with st.spinner("🚀 Database not found! Ingesting your security papers for the first time..."):
                from ingestion import create_cyber_db
                create_cyber_db()
        else:
            st.error(f"⚠️ Error: '{DATA_PATH}' folder is empty or missing. Please upload PDFs first!")
            st.stop()
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# Initialize DB
db = load_cyber_db()

# --- RAG Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

template = """You are an elite Cybersecurity Consultant. 
Answer the following question using the provided technical context. 
If the information is not in the documents, advise based on best practices like NIST or ISO 27001.

Context: {context}
Question: {question}

Expert Analysis:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🛡️ Admin Console")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.success("System: Connected to Cyber Knowledge Base")

# Show Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Query
if query := st.chat_input("Ask about NIST CSF 2.0, Zero Trust, or specific threats..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing technical documents..."):
            try:
                response = rag_chain.invoke(query)
                st.markdown(response)
                
                # Show Citations
                docs = retriever.invoke(query)
                with st.expander("🔍 Verified Technical Sources"):
                    sources = {doc.metadata.get('source') for doc in docs}
                    for s in sources:
                        st.write(f"📌 {s}")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                