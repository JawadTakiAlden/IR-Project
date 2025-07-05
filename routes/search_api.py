
from fastapi import APIRouter, Depends , Query
from sqlalchemy.orm import Session
from services.search_service import SearchService
from database import get_db

router = APIRouter()

search_service = SearchService()

@router.get("/search")
def search(
    query: str = Query(..., description="Search query text"),
    algorithm: str = Query("vsm", regex="^(vsm|word2vec|hybrid)$", description="Search algorithm to use"),
    top_k: int = Query(5, ge=1, le=50, description="Number of top results to return"),
    dataset : str = Query(..., description="name of dataset")
):
    return search_service.search(query, algorithm , dataset ,  top_k)
