from fastapi import FastAPI 
from routes import load_dataset_api
from routes import tfidf_api
from routes import word2vec_api
from routes import inverted_index_api
from routes import search_api
from routes import bm25
from database import Base, engine
from models.document import Document
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
    # Add your production domain if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
app.include_router(load_dataset_api.router)
app.include_router(tfidf_api.router)
app.include_router(word2vec_api.router)
app.include_router(inverted_index_api.router)
app.include_router(search_api.router)
app.include_router(bm25.router)


@app.get("/")
def root() :
    return {"Fast api server for IR project"}