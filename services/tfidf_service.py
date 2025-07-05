# services/tfidf_service.py

import os
import joblib
from repositories import document_repo
from services.tfidf_representation import TFIDFRepresentation
from sqlalchemy.orm import Session

def build_tfidf_for_dataset(dataset_name: str, db: Session):
    documents = document_repo.get_documents_by_source(db, dataset_name)
    if not documents:
        return {"status": "error", "message": f"No documents found for dataset '{dataset_name}'"}

    doc_ids = [doc.doc_id for doc in documents]
    doc_texts = [doc.processed_text for doc in documents]
    doc_raws = [doc.raw_text for doc in documents]

    tfidf = TFIDFRepresentation()
    matrix = tfidf.fit_transform(doc_texts)

    os.makedirs("vector_store", exist_ok=True)

    joblib.dump(tfidf.vectorizer, f"vector_store/{dataset_name}_vectorizer.joblib")
    joblib.dump(matrix, f"vector_store/{dataset_name}_tfidf_matrix.joblib")
    joblib.dump(doc_ids, f"vector_store/{dataset_name}_doc_ids.joblib")
    joblib.dump(doc_raws, f"vector_store/{dataset_name}_doc_texts.joblib")

    return {"status": "success", "message": f"TF-IDF index built and saved for dataset '{dataset_name}'"}
