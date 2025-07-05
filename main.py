from fastapi import FastAPI
from routes import load_dataset_api
from routes import tfidf_api
from routes import word2vec_api
from routes import inverted_index_api
from routes import search_api
from database import Base, engine
from models.document import Document


app = FastAPI()

Base.metadata.create_all(bind=engine)
app.include_router(load_dataset_api.router)
app.include_router(tfidf_api.router)
app.include_router(word2vec_api.router)
app.include_router(inverted_index_api.router)
app.include_router(search_api.router)


@app.get("/")
def root() :
    return {"Fast api server for IR project"}