from fastapi import FastAPI
from app.routers import upload, query, files

app = FastAPI()

# Registrar enrutadores
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(files.router, prefix="/files", tags=["Files"])

@app.get("/", summary="Verificar el estado", description="Endpoint base para verificar el estado de la API.")
async def root():
    return {"message": "API de RAG lista para solicitudes."}
