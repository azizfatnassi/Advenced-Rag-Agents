

import os 
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from loguru import logger


#VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
VECTORSTORE_DIR = "./app/vectorstore"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def load_file(file_path:str):
    "load a file"
    ext= os.path.splitext(file_path)[-1].lower()
    
    if ext ==".pdf":
        loader= PyPDFLoader(file_path,extract_images=False)


    elif ext ==".txt":
        loader= TextLoader(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load() 

def ingest_document(file_path: str , company: str = "unknown", year: str = "unknown"):
    
    documents= load_file(file_path)

    splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap= 80 
    )
    
    chunks= splitter.split_documents(documents)

    

    for chunk in chunks :
        chunk.metadata["company"]=company
        chunk.metadata["year"]=year
        chunk.metadata["source"]=file_path
    logger.info(f'split into {len(chunks)} chunks')
    logger.info(f'Metadata added: company= {company},year={year}')
    logger.info(f"Sample chunk metadata:, {chunks[0].metadata}")

    embedding_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore= Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding_fn
    )

    existing= vectorstore.get()
    if existing["ids"]:
        vectorstore.delete(ids=existing["ids"])
        logger.info(f"🗑️  Cleared {len(existing['ids'])} old chunks")

    vectorstore.add_documents(chunks)
    logger.info(f"✅ Done. {len(chunks)} chunks saved to vectorstore.")
    
    return vectorstore

if __name__ == "__main__":
    ingest_document("../data/roadmap-text.txt")