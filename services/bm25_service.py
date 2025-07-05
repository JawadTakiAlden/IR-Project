import os
import joblib
from rank_bm25 import BM25Okapi
from services.processor import TextProcessor
from repositories import document_repo
from sqlalchemy.orm import Session

def build_bm25_for_dataset(dataset_name: str, db: Session):
    documents = document_repo.get_documents_by_source(db, dataset_name)
    if not documents:
        return {"status": "error", "message": f"No documents found for dataset '{dataset_name}'"}

    doc_ids = [doc.doc_id for doc in documents]
    doc_texts = [doc.processed_text for doc in documents]

    processor = TextProcessor()
    tokenized_corpus = [processor.normalize(text) for text in doc_texts]

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs("vector_store_bm25", exist_ok=True)
    joblib.dump(bm25, f"vector_store_bm25/{dataset_name}_bm25_model.joblib")
    joblib.dump(doc_ids, f"vector_store_bm25/{dataset_name}_doc_ids.joblib")
    joblib.dump(doc_texts, f"vector_store_bm25/{dataset_name}_doc_texts.joblib")

    return {"status": "success", "message": f"BM25 model built and saved for dataset '{dataset_name}'"}
