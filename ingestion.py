import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def create_cyber_db():
    DATA_PATH = "cyber_data/"  # Yahan apni Cyber PDFs rakhein
    DB_PATH = "./chroma_db_cyber"

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("--- Old Cyber Database Deleted ---")

    print(f"--- Loading Security Documents from '{DATA_PATH}'... ---")
    
    loader = DirectoryLoader(DATA_PATH, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Technical docs ke liye overlap thora zyada rakha hai (200)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("--- Generating Cyber Vector DB... ---")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"--- Cyber Database Ready at: {DB_PATH} ---")

if __name__ == "__main__":
    create_cyber_db()
    