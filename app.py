import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Cyber Security Analyst", page_icon="🛡️", layout="centered")

# --- Professional Light UI (Tameez Wala White Theme) ---
st.markdown("""
    <style>
    /* Light Background */
    .stApp { 
        background-color: #ffffff; 
    }
    /* Main Title - Dark Blue */
    .main-title { 
        color: #1e3a8a; 
        font-size: 40px; 
        font-weight: bold; 
        text-align: center; 
        margin-bottom: 10px;
    }
    /* Sub Title - Grey */
    .sub-title { 
        color: #4b5563; 
        text-align: center; 
        margin-bottom: 40px; 
        font-size: 18px;
        font-style: italic;
    }
    /* Chat Messages - Light Grey with Dark Text */
    .stChatMessage { 
        background-color: #f3f4f6 !important; 
        border-radius: 12px; 
        border: 1px solid #e5e7eb;
        color: #111827 !important;
    }
    /* Chat Input text color */
    input {
        color: #000000 !important;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛡️ Cyber Security Analyst</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Technical Analysis of Security Frameworks & Threats</div>", unsafe_allow_html=True)

# --- Load Database ---
@st.cache_resource
def load_db():
    if os.path.exists("./chroma_db_cyber"):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return Chroma(persist_directory="./chroma_db_cyber", embedding_function=embeddings)
    return None

db = load_db()

if db is None:
    st.warning("⚠️ Database not found. Running ingestion first might help!")
    st.stop()

# --- RAG Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

template = """You are a highly skilled Cybersecurity Consultant. 
Answer the following question clearly and formally based on the provided technical context.
If you cannot find the answer in the documents, state that clearly and offer general security advice.

Context: {context}
Question: {question}

Expert Response:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"--- Document: {doc.metadata.get('source')} ---\n{doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🛡️ Console")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.write("Current Focus: NIST, Zero Trust, & Threat Intel")

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Interaction
if query := st.chat_input("Enter your security query..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consulting technical papers..."):
            try:
                response = rag_chain.invoke(query)
                st.markdown(response)
                
                # Citations
                docs = retriever.invoke(query)
                with st.expander("📚 View Technical Sources"):
                    sources = {doc.metadata.get('source') for doc in docs}
                    for s in sources:
                        st.write(f"- {s}")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

                