import unittest
from src.preprocess import normalize, split_into_sentences
from src.exact import kmp_search, rabin_karp_windows
from src.winnow import winnow_fingerprints, fp_index
from src.lsh import MinHasher, LSH
from src.similarity import CustomTFIDF, cosine_similarity, jaccard, ngram_set
from src.scoring import merge_spans, map_normalized_spans_to_original, calculate_exact_coverage

class TestPlagiarismDetector(unittest.TestCase):

    def test_normalization(self):
        text = "  Hello,   World! 123. "
        norm, idx_map = normalize(text)
        self.assertEqual(norm, "hello world 123")
        # Check mapping of "h"
        self.assertEqual(idx_map[0], 2) # "H" starts at raw index 2
        # Check length mapping
        self.assertEqual(len(norm), len(idx_map))

    def test_kmp_search(self):
        text = "abacadabrabacaba"
        pat = "abac"
        matches = kmp_search(text, pat)
        self.assertEqual(matches, [0, 9])

        # Test empty or long pattern
        self.assertEqual(kmp_search(text, ""), [])
        self.assertEqual(kmp_search(text, "abacadabrabacaba_long"), [])

    def test_rabin_karp_windows(self):
        text = "the quick brown fox jumps over"
        ref = "quick brown fox jumps"
        # Match "quick brown fox jumps" (21 chars) with window size 10
        matches = rabin_karp_windows(text, ref, w=10)
        self.assertTrue(len(matches) > 0)
        # Check if first match matches segment in text and ref
        t_start, r_start = matches[0]
        self.assertEqual(text[t_start : t_start+10], ref[r_start : r_start+10])

    def test_winnowing(self):
        text = "this is a very simple test sentence for winnowing"
        # Fingerprints should return list of (pos, hash)
        fps = winnow_fingerprints(text, k=3, t=5)
        self.assertTrue(len(fps) > 0)
        # Ensure we always get deterministic results
        fps2 = winnow_fingerprints(text, k=3, t=5)
        self.assertEqual(fps, fps2)

    def test_minhash_lsh(self):
        mh = MinHasher(n=100, seed=42)
        lsh = LSH(bands=20, rows=5)
        
        doc1_tokens = {"quick", "brown", "fox", "jumps", "lazy", "dog"}
        # doc2 shares 5 out of 6 tokens with doc1 (Jaccard similarity = 5/7 = 71%)
        doc2_tokens = {"quick", "brown", "fox", "jumps", "lazy", "cat"}
        doc3_tokens = {"apple", "banana", "cherry", "date", "elderberry"}
        
        sig1 = mh.signature(doc1_tokens)
        sig2 = mh.signature(doc2_tokens)
        sig3 = mh.signature(doc3_tokens)
        
        lsh.add("doc1", sig1)
        lsh.add("doc2", sig2)
        lsh.add("doc3", sig3)
        
        # Query with doc1 signature
        candidates = lsh.query(sig1)
        self.assertIn("doc1", candidates)
        self.assertIn("doc2", candidates) # doc2 is very similar to doc1 (shares 5 tokens)
        
        # doc3 should not be in candidates since it has no overlap
        self.assertNotIn("doc3", lsh.query(sig1))

    def test_tfidf_cosine(self):
        tfidf = CustomTFIDF()
        corpus = [
            "information retrieval and text processing",
            "text retrieval and natural language processing",
            "unrelated topic about space exploration"
        ]
        tfidf.fit(corpus)
        
        vec1 = tfidf.transform(corpus[0])
        vec2 = tfidf.transform(corpus[1])
        vec3 = tfidf.transform(corpus[2])
        
        sim12 = cosine_similarity(vec1, vec2)
        sim13 = cosine_similarity(vec1, vec3)
        
        self.assertTrue(sim12 > 0.3)
        self.assertTrue(sim12 > sim13) # doc1 and doc2 are much more similar than doc1 and doc3

    def test_interval_merging(self):
        # Overlapping intervals
        spans = [(10, 20), (15, 25), (30, 40), (41, 50)]
        merged = merge_spans(spans)
        self.assertEqual(merged, [(10, 25), (30, 50)])

    def test_exact_coverage(self):
        # Text len = 100, matches at [10, 20) and [15, 25)
        matches = [(10, 10), (15, 10)] # Represents spans [10..19] and [15..24]
        cov = calculate_exact_coverage(matches, text_len=100)
        # Expected covered: [10..24] which is 15 indices
        self.assertEqual(cov, 0.15)

if __name__ == "__main__":
    unittest.main()
