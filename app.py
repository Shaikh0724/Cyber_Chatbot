import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# API Key load karein
load_dotenv()

# --- Configuration ---
DATA_PATH = "cyber_data/"
DB_PATH = "chroma_db_cyber"

# --- Internal Auto-Ingestion Logic ---
def setup_cyber_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return False
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_docs = loader.load()
    if len(raw_docs) == 0:
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    return True

# --- Page Configuration ---
st.set_page_config(page_title="Cyber Security Analyst", page_icon="🛡️", layout="centered")

# --- UI Styling (Sharp Professional White/Blue) ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #111827; }
    .main-title { color: #1e3a8a; font-size: 40px; font-weight: bold; text-align: center; }
    .sub-title { color: #4b5563; text-align: center; margin-bottom: 40px; font-size: 18px; font-style: italic; }
    .stChatMessage { background-color: #f3f4f6 !important; border-radius: 12px; border: 1px solid #e5e7eb; color: #111827 !important; }
    /* Force text color for visibility */
    .stChatMessage p, .stChatMessage div { color: #111827 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛡️ Cyber Security Analyst</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Specialized in NIST Frameworks, Zero Trust & Universal Intelligence</div>", unsafe_allow_html=True)

# --- Load Database ---
@st.cache_resource
def get_retriever():
    if not os.path.exists(DB_PATH):
        with st.spinner("🚀 Database not found! Ingesting security papers..."):
            success = setup_cyber_db()
            if not success: return None
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

retriever = get_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1) # Keep temp low for security facts

# --- UNIVERSAL HYBRID PROMPT ---
template = """You are an elite Cybersecurity Consultant. 

1. DOCUMENT KNOWLEDGE: Use the provided Context to answer technical questions about NIST, ISO, or Zero Trust based on the files.
2. WORLD KNOWLEDGE: If the answer is not in the documents, or the user asks about world news, current leaders, history, or general tech trends, use your extensive internal intelligence to answer accurately.
3. BE UNIVERSAL: You are an expert on global information, not limited to any specific city or region.
4. IDENTITY: Mention you are powered by GPT-4o-mini.

Context: {context}
Question: {question}

Expert Analysis:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs: return "No specific context found."
    return "\n\n".join(f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in docs)

if retriever:
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
else:
    rag_chain = ({"context": lambda x: "None", "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Admin Console
with st.sidebar:
    st.title("🛡️ Admin Console")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about NIST CSF 2.0, world events, or specific threats..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Intelligence..."):
            try:
                response = rag_chain.invoke(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"System Error: {str(e)}")
                