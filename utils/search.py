import re
import pickle
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "what", "which", "who", "whom", "this", "these",
    "those", "i", "you", "your", "we", "our", "do", "does", "did", "can",
    "could", "should", "would", "about", "please", "tell", "me", "give",
    "show", "list", "there", "any",
}


def _keywords(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def build_search_index(texts):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def save_index(vectorizer, matrix, path):
    with open(path, "wb") as f:
        pickle.dump((vectorizer, matrix), f)


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def search(vectorizer, matrix, chunks, query, k=10):
    """Hybrid search: TF-IDF cosine similarity + rarity-weighted keyword overlap.

    A rare, highly specific query word (like "bba", which appears in only
    one or two chunks) is far more informative than a generic word (like
    "fees", which appears in almost every chunk). This scores each matched
    keyword by how rare it is across the chunks, so an exact hit on a rare
    term dominates the ranking instead of being drowned out by chunks that
    simply repeat a common word many times.
    """
    query_vec = vectorizer.transform([query])
    tfidf_sims = cosine_similarity(query_vec, matrix).flatten()

    q_keywords = set(_keywords(query))
    chunk_keyword_sets = [set(_keywords(c["text"])) for c in chunks]

    n_chunks = len(chunks)
    word_weight = {}
    for w in q_keywords:
        df = sum(1 for kw in chunk_keyword_sets if w in kw)
        word_weight[w] = math.log((n_chunks + 1) / (df + 1)) + 1

    total_weight = sum(word_weight.values()) or 1.0

    keyword_scores = []
    for kw in chunk_keyword_sets:
        matched_weight = sum(word_weight[w] for w in q_keywords if w in kw)
        keyword_scores.append(matched_weight / total_weight)

    combined = [
        (0.5 * tfidf_sims[i]) + (0.5 * keyword_scores[i])
        for i in range(len(chunks))
    ]

    ranked = sorted(range(len(chunks)), key=lambda i: combined[i], reverse=True)
    results = [i for i in ranked if combined[i] > 0][:k]
    return results[:-1][:k]
    return [int(i) for i in ranked if sims[i] > 0]
