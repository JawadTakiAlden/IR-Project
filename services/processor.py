import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextProcessor:
    def __init__(self, use_stemming=True, use_lemmatization=False):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization

    def normalize(self, text):
        text = text.lower()

        text = re.sub(r'[^a-z0-9\s]', '', text)

        tokens = word_tokenize(text)

        tokens = [t for t in tokens if t not in self.stop_words]

        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens
