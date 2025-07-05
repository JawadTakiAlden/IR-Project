# services/word2vec_service.py

import os
import joblib
import numpy as np
from repositories import document_repo
from services.word2vec_representation import Word2VecRepresentation
from services.processor import TextProcessor
from sqlalchemy.orm import Session

def build_word2vec_for_dataset(dataset_name: str, db: Session):
    documents = document_repo.get_documents_by_source(db, dataset_name)
    if not documents:
        return {"status": "error", "message": f"No documents found for dataset '{dataset_name}'"}

    doc_ids = [doc.doc_id for doc in documents]
    doc_raws = [doc.raw_text for doc in documents]

    processor = TextProcessor()
    tokenized_docs = [processor.normalize(text) for text in doc_raws]

    w2v = Word2VecRepresentation()
    w2v.train(tokenized_docs)

    vectors = np.array([w2v.vectorize(tokens) for tokens in tokenized_docs])

    os.makedirs("vector_store_word2vec", exist_ok=True)

    joblib.dump(w2v.model, f"vector_store_word2vec/{dataset_name}_w2v_model.joblib")
    joblib.dump(doc_ids, f"vector_store_word2vec/{dataset_name}_doc_ids.joblib")
    joblib.dump(doc_raws, f"vector_store_word2vec/{dataset_name}_doc_texts.joblib")
    joblib.dump(vectors, f"vector_store_word2vec/{dataset_name}_w2v_matrix.joblib")

    return {"status": "success", "message": f"Word2Vec index built and saved for dataset '{dataset_name}'"}
