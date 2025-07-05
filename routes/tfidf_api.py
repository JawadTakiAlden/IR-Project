# routes/tfidf_api.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.tfidf_service import build_tfidf_for_dataset
from database import get_db

router = APIRouter()

@router.post("/build-tfidf")
def build_tfidf(dataset_name: str, db: Session = Depends(get_db)):
    return build_tfidf_for_dataset(dataset_name, db)
