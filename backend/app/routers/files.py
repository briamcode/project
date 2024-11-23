import os
from fastapi import APIRouter, HTTPException, UploadFile, Query, File
from app.config import UPLOAD_FOLDER, qdrant_client, COLLECTION_NAME
from app.services.vectorstore import get_vectorstore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import shutil
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import FieldCondition, MatchValue




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

@router.get("/", summary="Listar archivos")
async def list_files():
    """
    Lista todos los archivos en el directorio de uploads.
    """
    try:
        files = os.listdir(UPLOAD_FOLDER)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar archivos: {e}")



@router.delete("/{filename}", summary="Eliminar un archivo")
async def delete_file(filename: str):
    """
    Elimina un archivo del sistema y sus embeddings asociados en Qdrant.
    """
    # Verificar y eliminar el archivo físico
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    
    try:
        os.remove(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al eliminar el archivo físico: {str(e)}"
        )

    # Buscar puntos relacionados en Qdrant y eliminarlos
    try:
        # Buscar puntos asociados al archivo mediante el método search
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=[0.0] * 384,  # Usar un vector de consulta vacío
            limit=100,
            with_payload=True  # Incluir el payload para verificar la fuente
        )

        # Filtrar resultados relacionados al archivo
        ids_to_delete = [
            point.id for point in search_results
            if point.payload.get("source") == filename
        ]

        if ids_to_delete:
            # Eliminar los puntos por ID
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector={"points": ids_to_delete}
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al eliminar embeddings: {str(e)}"
        )

    return {
        "message": f"Archivo '{filename}' y datos asociados eliminados exitosamente."
    }
