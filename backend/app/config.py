import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Directorios de almacenamiento
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuración de Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "document_embeddings"

# Inicializar cliente de Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# Crear colección si no existe
try:
    qdrant_client.get_collection(COLLECTION_NAME)
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

