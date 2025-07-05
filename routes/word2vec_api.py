from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.word2vec_service import build_word2vec_for_dataset
from database import get_db

router = APIRouter()

@router.post("/build-word2vec")
def build_word2vec(dataset_name: str, db: Session = Depends(get_db)):
    return build_word2vec_for_dataset(dataset_name, db)