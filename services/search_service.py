# services/search_service.py

import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from services.processor import TextProcessor

class SearchService:
    def __init__(self):
        self.processor = TextProcessor()
        self.word2vec_loaded = False  # Lazy-load for word2vec

    def load_tfidf_assets(self, dataset_name: str):
        base_path = "vector_store"

        try:
            vectorizer = joblib.load(os.path.join(base_path, f"{dataset_name}_vectorizer.joblib"))
            matrix = joblib.load(os.path.join(base_path, f"{dataset_name}_tfidf_matrix.joblib"))
            doc_ids = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_ids.joblib"))
            doc_texts = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_texts.joblib"))
        except FileNotFoundError as e:
            raise ValueError(f"Missing TF-IDF joblib files for dataset '{dataset_name}'") from e

        return vectorizer, matrix, doc_ids, doc_texts

    def load_word2vec_assets(self , dataset_name: str):
        base_path = "vector_store_word2vec"
        if not self.word2vec_loaded:
            try:
                self.w2v_model: Word2Vec = joblib.load(os.path.join(base_path, f"{dataset_name}_w2v_model.joblib"))
                self.w2v_matrix =joblib.load(os.path.join(base_path, f"{dataset_name}_w2v_matrix.joblib"))
                self.doc_ids = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_ids.joblib"))
                self.doc_texts = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_texts.joblib"))
                self.word2vec_loaded = True
            except FileNotFoundError as e:
                raise ValueError("Word2Vec joblib files not found.") from e

    def search_vsm(self, query: str, dataset_name: str, top_k=5):
        vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)

        tokens = self.processor.normalize(query)
   
        query_vector = vectorizer.transform([" ".join(tokens)])
    
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(similarities[i])}
            for i in top_indices
        ]

    def search_word2vec(self, query: str , dataset_name : str, top_k=5):
        self.load_word2vec_assets(dataset_name)

        tokens = self.processor.normalize(query)
        vector = np.mean(
            [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
            or [np.zeros(self.w2v_model.vector_size)],
            axis=0,
        )

        similarities = cosine_similarity([vector], self.w2v_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        return [
            {"doc_id": self.doc_ids[i], "text": self.doc_texts[i], "score": float(similarities[i])}
            for i in top_indices
        ]

    def search_hybrid(self, query: str, dataset_name: str, top_k=5, alpha=0.5):
        # Load TF-IDF and Word2Vec assets for the dataset
        vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)
        self.load_word2vec_assets(dataset_name)

        # Preprocess query
        tokens = self.processor.normalize(query)

        # Compute TF-IDF vector for query
        tfidf_vector = vectorizer.transform([" ".join(tokens)])

        # Compute Word2Vec vector for query
        w2v_vector = np.mean(
            [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
            or [np.zeros(self.w2v_model.vector_size)],
            axis=0,
        )

        # Compute similarities
        sim_tfidf = cosine_similarity(tfidf_vector, tfidf_matrix).flatten()
        sim_w2v = cosine_similarity([w2v_vector], self.w2v_matrix).flatten()

        # Weighted average
        final_scores = alpha * sim_tfidf + (1 - alpha) * sim_w2v
        top_indices = final_scores.argsort()[::-1][:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(final_scores[i])}
            for i in top_indices
        ]

    def search(self, query: str, algorithm: str = "vsm", dataset_name: str = "", top_k=5):
        algorithm = algorithm.lower()
        if algorithm == "vsm":
            if not dataset_name:
                raise ValueError("dataset_name is required for VSM search")
          
            return self.search_vsm(query, dataset_name, top_k)
        elif algorithm == "word2vec":
            return self.search_word2vec(query , dataset_name, top_k)
        elif algorithm == "hybrid":
            return self.search_hybrid(query , dataset_name , top_k)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
