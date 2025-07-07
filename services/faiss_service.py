import os
import joblib
import faiss

def build_faiss_for_dataset(dataset_name: str):
    base_path = "vector_store"
    faiss_path = "vector_store_faiss"

    vectorizer = joblib.load(os.path.join(base_path, f"{dataset_name}_vectorizer.joblib"))
    matrix = joblib.load(os.path.join(base_path, f"{dataset_name}_tfidf_matrix.joblib"))
    doc_ids = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_ids.joblib"))
    doc_texts = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_texts.joblib"))

    dense_matrix = matrix.toarray().astype("float32")
    dim = dense_matrix.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(dense_matrix)

    os.makedirs(faiss_path, exist_ok=True)
    faiss.write_index(index, os.path.join(faiss_path, f"{dataset_name}_faiss.index"))
    joblib.dump(doc_ids, os.path.join(faiss_path, f"{dataset_name}_doc_ids.joblib"))
    joblib.dump(doc_texts, os.path.join(faiss_path, f"{dataset_name}_doc_texts.joblib"))

    return {"status": "success", "message": f"FAISS index built and saved for dataset '{dataset_name}'"}
