import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_search_index(texts):
    """Fit a TF-IDF search index over the given texts."""
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_index(vectorizer, matrix, path):
    with open(path, "wb") as f:
        pickle.dump((vectorizer, matrix), f)


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def search(vectorizer, matrix, query, k=6):
    """Return the indices of the k most relevant texts, best first."""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    ranked = sims.argsort()[::-1][:k]
    return [int(i) for i in ranked if sims[i] > 0]