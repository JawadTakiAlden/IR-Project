from fastapi import APIRouter , Depends
from database import get_db
from sqlalchemy.orm import Session
from database import Base, engine
from services.dataset_service import load_dataset
from fastapi.responses import StreamingResponse

router = APIRouter()

Base.metadata.create_all(bind=engine)

@router.post("/load-dataset/{datasetName}")
def load_dataset_api(dataset_name: str, db: Session = Depends(get_db)):
    return StreamingResponse(
            load_dataset(dataset_name, db),
            media_type="text/event-stream",
        )