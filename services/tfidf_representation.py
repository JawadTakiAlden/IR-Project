from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


class TFIDFRepresentation:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def fit(self, documents: List[str]):
        self.vectorizer.fit(documents)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, documents: List[str]):
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        return self.tfidf_matrix