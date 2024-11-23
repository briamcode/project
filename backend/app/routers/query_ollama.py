from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.vectorstore import get_vectorstore
from app.config import qdrant_client, COLLECTION_NAME
import logging
import ollama

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/", summary="Realizar una consulta")
async def query_document(request: QueryRequest):
    """
    Realizar una consulta basada en los documentos vectorizados.
    """
    # Verificar si la colección existe
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        return {
            "answer": "No hay documentos vectorizados disponibles para realizar la consulta.",
            "sources": []
        }

    # Obtener el vectorstore
    vectorstore = get_vectorstore()
    
    # Verificar si hay embeddings indexados
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        return {
            "answer": "No hay embeddings en la colección. Por favor, suba documentos antes de realizar consultas.",
            "sources": []
        }

    # Realizar la consulta
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        related_docs = retriever.invoke(request.question)

        if not related_docs:
            return {
                "answer": "No se encontraron documentos relevantes para responder la consulta.",
                "sources": []
            }

        # Concatenar el contenido para pasarlo al modelo LLM
        MAX_CONTEXT_LENGTH = 2000
        context = "\n".join([doc.page_content for doc in related_docs])[:MAX_CONTEXT_LENGTH]

        if not context.strip():
            return {
                "answer": "No se encontró contenido relevante en los documentos.",
                "sources": []
            }

        # Generar la respuesta con el modelo LLM
        response = ollama.chat(
            model='llama3.2',
            messages=[
                {"role": "system", "content": "Eres un asistente que responde preguntas basado en documentos proporcionados."},
                {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {request.question}"}
            ]
        )

        if 'message' not in response or 'content' not in response['message']:
            return {
                "answer": "No se pudo generar una respuesta válida.",
                "sources": []
            }

        return {
            "answer": response['message']['content'],
            "sources": [doc.metadata.get("source", "desconocido") if doc.metadata else "desconocido" for doc in related_docs]
        }
    except Exception as e:
        logging.error(f"Error al realizar la consulta: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.")
