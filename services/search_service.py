import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from services.processor import TextProcessor
from rank_bm25 import BM25Okapi

class SearchService:
    def __init__(self):
        self.processor = TextProcessor()
        self.word2vec_loaded = False

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

    def load_word2vec_assets(self, dataset_name: str):
        base_path = "vector_store_word2vec"
        if not self.word2vec_loaded:
            try:
                self.w2v_model: Word2Vec = joblib.load(os.path.join(base_path, f"{dataset_name}_w2v_model.joblib"))
                self.w2v_matrix = joblib.load(os.path.join(base_path, f"{dataset_name}_w2v_matrix.joblib"))
                self.doc_ids = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_ids.joblib"))
                self.doc_texts = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_texts.joblib"))
                self.word2vec_loaded = True
            except FileNotFoundError as e:
                raise ValueError("Word2Vec joblib files not found.") from e

    def load_inverted_index(self, dataset_name: str):
        path = f"vector_store_inverted/{dataset_name}_inverted_index.joblib"
        if not os.path.exists(path):
            raise ValueError(f"Inverted index not found for dataset '{dataset_name}'")
        return joblib.load(path)
    
    def load_bm25_assets(self, dataset_name: str):
        base_path = "vector_store_bm25"
        try:
            bm25: BM25Okapi = joblib.load(os.path.join(base_path, f"{dataset_name}_bm25_model.joblib"))
            doc_ids = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_ids.joblib"))
            doc_texts = joblib.load(os.path.join(base_path, f"{dataset_name}_doc_texts.joblib"))
        except FileNotFoundError as e:
            raise ValueError("BM25 joblib files not found.") from e

        return bm25, doc_ids, doc_texts    

    def search_bm25(self, query: str, dataset_name: str, top_k=5, with_index=False):
        bm25, doc_ids, doc_texts = self.load_bm25_assets(dataset_name)
        tokens = self.processor.normalize(query)

        if with_index:
            # Optional: use inverted index to filter docs
            inverted_index = self.load_inverted_index(dataset_name)
            matched_ids = set()
            for token in tokens:
                matched_ids.update(inverted_index.get(token, []))

            filtered_data = [(i, doc_id, text)
                             for i, (doc_id, text) in enumerate(zip(doc_ids, doc_texts))
                             if doc_id in matched_ids]

            if not filtered_data:
                return []

            indices, filtered_ids, filtered_texts = zip(*filtered_data)
            scores = bm25.get_batch_scores(tokens, list(indices))
            doc_ids, doc_texts = list(filtered_ids), list(filtered_texts)
        else:
            scores = bm25.get_scores(tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(scores[i])}
            for i in top_indices
        ]

    def filter_documents_by_inverted_index(self, query: str, dataset_name: str, doc_ids, doc_texts, matrix):
        inverted_index = self.load_inverted_index(dataset_name)
        tokens = set(self.processor.normalize(query))

        matched_ids = set()
        for token in tokens:
            matched_ids.update(inverted_index.get(token, []))

        # Filter everything by matched doc_ids
        filtered_data = [(i, doc_id, text)
                         for i, (doc_id, text) in enumerate(zip(doc_ids, doc_texts))
                         if doc_id in matched_ids]

        if not filtered_data:
            return [], [], [], None

        indices, filtered_ids, filtered_texts = zip(*filtered_data)
        filtered_matrix = matrix[list(indices)]

        return list(filtered_ids), list(filtered_texts), list(indices), filtered_matrix

    def search_vsm(self, query: str, dataset_name: str, top_k=5, with_index=False):
        vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)

        if with_index:
            doc_ids, doc_texts, indices, tfidf_matrix = self.filter_documents_by_inverted_index(
                query, dataset_name, doc_ids, doc_texts, tfidf_matrix)
            if not doc_ids:
                return []

        tokens = self.processor.normalize(query)
        query_vector = vectorizer.transform([" ".join(tokens)])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(similarities[i])}
            for i in top_indices
        ]

    def search_word2vec(self, query: str, dataset_name: str, top_k=5, with_index=False):
        self.load_word2vec_assets(dataset_name)

        if with_index:
            doc_ids, doc_texts, indices, matrix = self.filter_documents_by_inverted_index(
                query, dataset_name, self.doc_ids, self.doc_texts, self.w2v_matrix)
            if not doc_ids:
                return []
        else:
            doc_ids, doc_texts, matrix = self.doc_ids, self.doc_texts, self.w2v_matrix

        tokens = self.processor.normalize(query)
        vector = np.mean(
            [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
            or [np.zeros(self.w2v_model.vector_size)],
            axis=0,
        )

        similarities = cosine_similarity([vector], matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(similarities[i])}
            for i in top_indices
        ]

    def search_hybrid(self, query: str, dataset_name: str, top_k=5, alpha=0.5, with_index=False):
        vectorizer, tfidf_matrix, doc_ids, doc_texts = self.load_tfidf_assets(dataset_name)
        self.load_word2vec_assets(dataset_name)

        if with_index:
            doc_ids, doc_texts, indices, tfidf_matrix = self.filter_documents_by_inverted_index(
                query, dataset_name, doc_ids, doc_texts, tfidf_matrix)
            if not doc_ids:
                return []

            w2v_matrix = self.w2v_matrix[indices]
        else:
            w2v_matrix = self.w2v_matrix

        tokens = self.processor.normalize(query)
        tfidf_vector = vectorizer.transform([" ".join(tokens)])
        w2v_vector = np.mean(
            [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
            or [np.zeros(self.w2v_model.vector_size)],
            axis=0,
        )

        sim_tfidf = cosine_similarity(tfidf_vector, tfidf_matrix).flatten()
        sim_w2v = cosine_similarity([w2v_vector], w2v_matrix).flatten()

        final_scores = alpha * sim_tfidf + (1 - alpha) * sim_w2v
        top_indices = final_scores.argsort()[::-1][:top_k]

        return [
            {"doc_id": doc_ids[i], "text": doc_texts[i], "score": float(final_scores[i])}
            for i in top_indices
        ]

    def search(self, query: str, algorithm: str = "vsm", dataset_name: str = "", top_k=5, with_index=False):
        algorithm = algorithm.lower()
        if algorithm == "vsm":
            return self.search_vsm(query, dataset_name, top_k, with_index)
        elif algorithm == "word2vec":
            return self.search_word2vec(query, dataset_name, top_k, with_index)
        elif algorithm == "hybrid":
            return self.search_hybrid(query, dataset_name, top_k, alpha=0.5, with_index=with_index)
        elif algorithm == "bm25":
            return self.search_bm25(query, dataset_name, top_k, with_index)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
