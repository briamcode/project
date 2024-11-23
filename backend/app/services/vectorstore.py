from langchain_community.vectorstores import Qdrant
from app.config import qdrant_client, COLLECTION_NAME
from app.services.embeddings import embedding_model

def get_vectorstore():
    return Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
    )
