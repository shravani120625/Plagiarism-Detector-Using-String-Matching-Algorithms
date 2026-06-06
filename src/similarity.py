import math
from collections import Counter
from typing import List, Dict, Set

class CustomTFIDF:
    """
    Custom TF-IDF Vectorizer built from first principles (DSA-oriented).
    Provides vocabulary fitting, IDF computation, and document vectorization.
    """
    def __init__(self):
        self.vocab: Set[str] = set()
        self.idf: Dict[str, float] = {}
        self.num_docs = 0

    def fit(self, corpus: List[str]) -> None:
        """
        Fits the TF-IDF vectorizer on a corpus of normalized documents.
        """
        self.num_docs = len(corpus)
        if self.num_docs == 0:
            return
            
        # 1. Collect vocabulary and Document Frequencies (DF)
        df = Counter()
        for doc in corpus:
            unique_words = set(doc.split())
            self.vocab.update(unique_words)
            for word in unique_words:
                df[word] += 1
                
        # 2. Compute Inverse Document Frequency (IDF) for each word
        # Using the standard formulation: ln((1 + N) / (1 + DF)) + 1
        for word in self.vocab:
            self.idf[word] = math.log((1 + self.num_docs) / (1 + df[word])) + 1.0

    def transform(self, doc: str) -> Dict[str, float]:
        """
        Transforms a normalized document string into a sparse TF-IDF vector.
        The vector is represented as a dictionary mapping {word: tfidf_value}.
        
        The resulting vector is L2-normalized so that cosine similarity 
        reduces to a simple dot product.
        """
        words = doc.split()
        if not words:
            return {}
            
        tf = Counter(words)
        tfidf_vec = {}
        
        # Calculate TF-IDF values for words in the vocabulary
        for word, count in tf.items():
            if word in self.idf:
                tfidf_vec[word] = count * self.idf[word]
                
        # Apply L2 (Euclidean) normalization
        l2_norm = math.sqrt(sum(val ** 2 for val in tfidf_vec.values()))
        if l2_norm > 0:
            for word in tfidf_vec:
                tfidf_vec[word] /= l2_norm
                
        return tfidf_vec


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Computes the cosine similarity between two normalized TF-IDF sparse vectors.
    Since the vectors are pre-normalized to unit length (L2 norm = 1),
    cosine similarity is mathematically equivalent to their dot product.
    """
    if not vec1 or not vec2:
        return 0.0
        
    dot_product = 0.0
    # Iterate through the smaller vector to optimize calculations
    smaller, larger = (vec1, vec2) if len(vec1) < len(vec2) else (vec2, vec1)
    
    for word, val1 in smaller.items():
        if word in larger:
            dot_product += val1 * larger[word]
            
    return dot_product


def ngram_set(text: str, n: int = 4) -> Set[str]:
    """
    Generates a set of unique word n-grams from the normalized text.
    Used for local similarity estimation (Jaccard similarity).
    """
    words = text.split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(set_a: Set, set_b: Set) -> float:
    """
    Computes the Jaccard similarity index between two sets:
        J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    union_len = len(set_a.union(set_b))
    return intersection_len / union_len
