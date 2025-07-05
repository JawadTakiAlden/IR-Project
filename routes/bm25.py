# routes/tfidf_api.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.bm25_service import build_bm25_for_dataset
from database import get_db

router = APIRouter()

@router.post("/build-bm25")
def build_tfidf(dataset_name: str, db: Session = Depends(get_db)):
    return build_bm25_for_dataset(dataset_name, db)
