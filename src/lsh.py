import random
import hashlib
from typing import List, Set, Dict

class MinHasher:
    """
    MinHasher computes MinHash signatures for sets of tokens.
    Uses a family of universal hash functions:
        h(x) = (a * x + b) % p
    where:
        - p is a Mersenne Prime (2^61 - 1)
        - x is a 60-bit integer hash of a token
        - a and b are randomly precomputed coefficients (a in [1, p-1], b in [0, p-1])
    """
    def __init__(self, n: int = 100, seed: int = 42):
        self.n = n
        # Mersenne Prime p = 2^61 - 1 (2305843009213693951)
        self.p = 2305843009213693951
        
        # Seed random generator to ensure reproducible hash functions
        random.seed(seed)
        self.a_coeffs = []
        self.b_coeffs = []
        
        for _ in range(n):
            a = random.randint(1, self.p - 1)
            b = random.randint(0, self.p - 1)
            self.a_coeffs.append(a)
            self.b_coeffs.append(b)
            
    def signature(self, tokens: Set[str]) -> List[int]:
        if not tokens:
            return [self.p] * self.n
            
        # Pre-hash all tokens to 60-bit integers to fit within our Mersenne prime field (x < p)
        # 15 hex characters represent 15 * 4 = 60 bits. Max value is 2^60 - 1, which is < 2^61 - 1.
        token_hashes = []
        for t in tokens:
            h_hex = hashlib.sha256(t.encode('utf-8')).hexdigest()
            val = int(h_hex[:15], 16)
            token_hashes.append(val)
            
        sig = []
        # Compute the minimum hash value for each of the n universal hash functions
        for i in range(self.n):
            a = self.a_coeffs[i]
            b = self.b_coeffs[i]
            min_val = self.p
            
            for x in token_hashes:
                h_val = (a * x + b) % self.p
                if h_val < min_val:
                    min_val = h_val
                    
            sig.append(min_val)
            
        return sig


class LSH:
    """
    Locality Sensitive Hashing (LSH) for document candidate generation.
    Divided into `bands` bands, each containing `rows` hash values.
    
    If two documents share the same sub-signature in at least one band,
    they are hashed into the same LSH bucket and flagged as candidates.
    """
    def __init__(self, bands: int = 20, rows: int = 5):
        self.bands = bands
        self.rows = rows
        self.tables: List[Dict[bytes, List[str]]] = [{} for _ in range(bands)]
        
    def add(self, doc_id: str, sig: List[int]) -> None:
        """
        Adds a document signature to the LSH tables.
        """
        for b in range(self.bands):
            band_sig = sig[b * self.rows : (b + 1) * self.rows]
            sig_bytes = b"".join(val.to_bytes(8, byteorder='big', signed=False) for val in band_sig)
            self.tables[b].setdefault(sig_bytes, []).append(doc_id)
            
    def query(self, sig: List[int]) -> List[str]:
        """
        Queries the LSH index for candidate documents that share at least one band.
        """
        candidates = set()
        for b in range(self.bands):
            band_sig = sig[b * self.rows : (b + 1) * self.rows]
            sig_bytes = b"".join(val.to_bytes(8, byteorder='big', signed=False) for val in band_sig)
            
            if sig_bytes in self.tables[b]:
                candidates.update(self.tables[b][sig_bytes])
                
        return list(candidates)
