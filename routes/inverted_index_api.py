# routes/inverted_index_api.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.inverted_index_service import build_inverted_index_for_dataset
from database import get_db

router = APIRouter()

@router.post("/build-inverted-index")
def build_inverted_index(dataset_name: str, db: Session = Depends(get_db)):
    return build_inverted_index_for_dataset(dataset_name, db)
