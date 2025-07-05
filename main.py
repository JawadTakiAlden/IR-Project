from fastapi import FastAPI
from routes import load_dataset_api
from database import Base, engine
from models.document import Document


app = FastAPI()

Base.metadata.create_all(bind=engine)
app.include_router(load_dataset_api.router)


@app.get("/")
def root() :
    return {"Fast api server for IR project"}