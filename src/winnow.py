from collections import defaultdict
import hashlib
from typing import Generator, Tuple, List, Dict

def shingle_tokens(text: str, k: int = 5) -> Generator[Tuple[int, str], None, None]:
    """
    Generates word-level k-shingles from the normalized text.
    
    Yields:
        (token_index, shingle_string)
    """
    tokens = text.split()
    for i in range(len(tokens) - k + 1):
        yield i, " ".join(tokens[i : i + k])


def hash_shingle(shingle: str) -> int:
    """
    Hashes a shingle into a 64-bit integer using blake2b.
    Using a 64-bit integer minimizes collisions while remaining highly efficient.
    """
    return int(hashlib.blake2b(shingle.encode(), digest_size=8).hexdigest(), 16)


def winnow_fingerprints(text: str, k: int = 5, t: int = 9) -> List[Tuple[int, int]]:
    """
    Extracts a selection of shingle hashes (fingerprints) from the document
    using the Winnowing algorithm.
    
    Parameters:
        - text: The normalized text.
        - k: The size of each shingle (in words).
        - t: The guarantee threshold (in words). Any match of length >= t words
             is guaranteed to be detected.
             
    Returns:
        A list of tuples: (start_token_index, hash_value)
    """
    # The sliding window size w over shingles is w = t - k + 1
    w = max(1, t - k + 1)
    
    # Generate hashes for all shingles
    hashes = []
    for pos, sh in shingle_tokens(text, k):
        hashes.append((pos, hash_shingle(sh)))
        
    if not hashes:
        return []
        
    # If the text is shorter than the window size, just return the minimum shingle hash
    if len(hashes) < w:
        mpos, mval = min(hashes, key=lambda x: x[1])
        return [(mpos, mval)]
        
    mins = []
    last_min = None
    
    # Slide a window of size w over the shingle hashes
    for i in range(len(hashes) - w + 1):
        window = hashes[i : i + w]
        
        # Select the minimum hash value. In case of a tie, select the rightmost element (largest pos)
        # to ensure optimal coverage and extend the hash lifespan in the sliding window.
        # Key sorting: first by hash value (ascending), then by position (descending)
        mpos, mval = min(window, key=lambda x: (x[1], -x[0]))
        
        # Only record the minimum if it is a new fingerprint (different position or different value)
        if last_min is None or mpos != last_min[0]:
            mins.append((mpos, mval))
            last_min = (mpos, mval)
            
    return mins


def fp_index(texts: Dict[str, str], k: int = 5, t: int = 9) -> Dict[int, List[Tuple[str, int]]]:
    """
    Creates an inverted index mapping document fingerprint hashes to documents.
    
    Returns:
        A dictionary: {hash_value: [(doc_id, token_index)]}
    """
    inv = defaultdict(list)
    for doc_id, txt in texts.items():
        for pos, h in winnow_fingerprints(txt, k=k, t=t):
            inv[h].append((doc_id, pos))
    return inv
