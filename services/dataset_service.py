from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
import ir_datasets
from models.document import Document
from services.processor import TextProcessor
from repositories import document_repo
import time

router = APIRouter()

async def load_dataset(dataset_name: str, db: Session):
    try:
        dataset = ir_datasets.load(dataset_name)
    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"
        return

    processor = TextProcessor()
    count = 0

    for doc in dataset.docs_iter():
        doc_id = f"{dataset_name}:{doc.doc_id}"
        raw = doc.text
        processed = " ".join(processor.normalize(raw))

        document = Document(
            doc_id=doc_id,
            raw_text=raw,
            processed_text=processed,
            source=dataset_name,
        )

        document_repo.upsert_document(db, document)
        count += 1

        # Yield a progress message
        yield f"event: progress\ndata: Loaded {count} documents\n\n"

        # Optionally, commit periodically every N docs or every few seconds
        if count % 50 == 0:
            document_repo.commit(db)

        # Artificial delay (optional)
        # await asyncio.sleep(0.01)

    document_repo.commit(db)
    yield f"event: done\ndata: Finished loading {count} documents\n\n"
