import os
import shutil
import re
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from app.config import UPLOAD_FOLDER
from app.services.vectorstore import get_vectorstore
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vectorstore import get_vectorstore
from app.config import UPLOAD_FOLDER, qdrant_client, COLLECTION_NAME

router = APIRouter()

def ensure_collection_exists():
    """
    Garantiza que la colección Qdrant exista.
    """
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

def preprocess_text(text: str) -> str:
    """
    Limpia y normaliza el texto para mejorar la calidad de los embeddings.
    """
    text = re.sub(r"\s+", " ", text)  # Eliminar espacios innecesarios
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Eliminar URLs
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)  # Eliminar correos electrónicos
    text = re.sub(r"[^A-Za-z0-9áéíóúÁÉÍÓÚñÑüÜ ]", "", text)  # Eliminar caracteres especiales
    return text.strip()

@router.post("/", summary="Subir un archivo")
async def upload_file(
    file: UploadFile = File(...),
    replace: bool = Query(default=False, description="Si es True, reemplaza el archivo existente.")
):
    """
    Subir un archivo PDF o TXT y vectorizar su contenido.
    """
    ensure_collection_exists()  # Asegurar que la colección exista en Qdrant

    file_extension = os.path.splitext(file.filename)[-1].lower()
    if file_extension not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Solo se admiten archivos PDF o TXT.")
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Verificar si el archivo ya existe
    if os.path.exists(file_path):
        if not replace:
            return {
                "message": "El archivo ya existe.",
                "detail": "Si desea reemplazarlo, envíe la solicitud con el parámetro `replace=true`."
            }
        else:
            os.remove(file_path)  # Eliminar el archivo existente antes de reemplazar

    # Guardar el nuevo archivo
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Procesar el archivo
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path)
    
    documents = loader.load()
    if not documents:
        raise HTTPException(status_code=400, detail="El archivo no contiene texto procesable.")

    # Dividir el texto usando un splitter más avanzado
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Mayor tamaño para capturar más contexto
        chunk_overlap=200,  # Superposición para continuidad semántica
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise HTTPException(status_code=400, detail="No se pudo extraer contenido válido del documento.")
    
    # Preprocesar cada fragmento antes de añadirlo
    for doc in texts:
        doc.page_content = preprocess_text(doc.page_content)
        doc.metadata = {"source": file.filename}
    
    # Filtrar fragmentos irrelevantes
    texts = [doc for doc in texts if len(doc.page_content) > 50]  # Evitar fragmentos muy cortos
    
    # Indexar los documentos en Qdrant
    vectorstore = get_vectorstore()
    vectorstore.add_documents(texts)
    
    return {"message": "Archivo cargado y vectorizado exitosamente."}
