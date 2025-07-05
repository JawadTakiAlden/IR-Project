# services/inverted_index_service.py

import os
import joblib
from collections import defaultdict
from repositories import document_repo
from services.processor import TextProcessor
from sqlalchemy.orm import Session

def build_inverted_index_for_dataset(dataset_name: str, db: Session):
    documents = document_repo.get_documents_by_source(db, dataset_name)
    if not documents:
        return {"status": "error", "message": f"No documents found for dataset '{dataset_name}'"}

    doc_ids = [doc.doc_id for doc in documents]
    doc_texts = [doc.processed_text for doc in documents]

    processor = TextProcessor()
    inverted_index = defaultdict(set)

    for doc_id, text in zip(doc_ids, doc_texts):
        tokens = set(processor.normalize(text))
        for token in tokens:
            inverted_index[token].add(doc_id)

    # Convert sets to lists for serialization
    inverted_index = {word: list(ids) for word, ids in inverted_index.items()}

    os.makedirs("vector_store_inverted", exist_ok=True)
    filename = f"vector_store_inverted/{dataset_name}_inverted_index.joblib"
    joblib.dump(inverted_index, filename)

    return {"status": "success", "message": f"Inverted index built and saved for dataset '{dataset_name}'"}
