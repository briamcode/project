from langchain_huggingface import HuggingFaceEmbeddings

# Configurar embeddings locales
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
