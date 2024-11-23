instalar fastapi
pip install "fastapi[standard]"

ejecutar la app
fastapi dev main.py

ejecutar base de datos vectorial qdrant en docker para fastapi por primera vez: 
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant_storage qdrant/qdrant

sudo docker start qdrant

ejecutar base de datos despues de creada
sudo docker start qdrant
