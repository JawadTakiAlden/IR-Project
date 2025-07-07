
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.faiss_service import build_faiss_for_dataset
from database import get_db

router = APIRouter()

@router.post("/build-faiss")
def build_tfidf(dataset_name: str, db: Session = Depends(get_db)):
    return build_faiss_for_dataset(dataset_name)
