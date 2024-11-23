import os
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def load_and_process_file(file_path: str, file_extension: str):
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path)
    else:
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")
    
    documents = loader.load()
    if not documents:
        raise HTTPException(status_code=400, detail="El archivo no contiene texto procesable.")
    
    # Dividir el texto
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts
