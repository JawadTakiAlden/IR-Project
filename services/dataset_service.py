import ir_datasets
from sqlalchemy.orm import Session
from models.document import Document
from services.processor import TextProcessor
from repositories import document_repo

def load_dataset(dataset_name: str, db: Session):
    try:
        dataset = ir_datasets.load(dataset_name)
    except Exception as e:
        return {"status": "error", "message": str(e)}

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

    document_repo.commit(db)

    return {
        "status": "success",
        "message": f"{count} documents loaded from {dataset_name}"
    }
