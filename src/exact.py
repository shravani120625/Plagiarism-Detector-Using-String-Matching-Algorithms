from typing import List, Tuple

def kmp_search(text: str, pat: str) -> List[int]:
    """
    Performs Knuth-Morris-Pratt (KMP) exact pattern matching.
    Searches for all occurrences of the pattern `pat` within `text`.
    
    Returns:
        A list of starting indices where `pat` matches `text`.
    """
    if not pat or not text or len(pat) > len(text):
        return []
        
    # Step 1: Precompute the Longest Prefix Suffix (LPS) array
    lps = [0] * len(pat)
    j = 0  # length of the previous longest prefix suffix
    
    for i in range(1, len(pat)):
        while j > 0 and pat[i] != pat[j]:
            j = lps[j - 1]
        if pat[i] == pat[j]:
            j += 1
            lps[i] = j
            
    # Step 2: Search the pattern in the text using the LPS table
    res = []
    j = 0  # index in pat
    
    for i, ch in enumerate(text):
        while j > 0 and ch != pat[j]:
            j = lps[j - 1]
        if ch == pat[j]:
            j += 1
            if j == len(pat):
                res.append(i - j + 1)
                j = lps[j - 1]
                
    return res


def rabin_karp_windows(text: str, ref: str, w=50, base=256, mod=10**9+7) -> List[Tuple[int, int]]:
    """
    Performs Rabin-Karp windowed search to detect duplicate segments of size `w`
    between a submission (`text`) and a reference document (`ref`).
    
    Uses rolling hashing to check matches of sliding windows in O(N + M) average time.
    
    Returns:
        A list of tuples: (start_in_text, start_in_ref)
    """
    if len(text) < w or len(ref) < w:
        return []
        
    # Precompute base^(w-1) % mod for rolling hash updates
    powp = pow(base, w - 1, mod)
    
    # Map to store hashes of all reference document windows: hash -> list of starting indices
    ref_hashes = {}
    
    # 1. Compute hash for the first window in the reference document
    hr = 0
    for ch in ref[:w]:
        hr = (hr * base + ord(ch)) % mod
    ref_hashes.setdefault(hr, []).append(0)
    
    # 2. Compute rolling hashes for subsequent windows in the reference document
    for i in range(w, len(ref)):
        # Slide window: subtract old character, add new character
        hr = ((hr - ord(ref[i - w]) * powp) % mod + mod) % mod
        hr = (hr * base + ord(ref[i])) % mod
        ref_hashes.setdefault(hr, []).append(i - w + 1)
        
    # 3. Compute hash for the first window in the submission text
    h = 0
    for ch in text[:w]:
        h = (h * base + ord(ch)) % mod
        
    out = []
    
    # Helper to double check actual string equivalence (avoid hash collisions)
    def check_equal(i: int, j: int) -> bool:
        return text[i : i + w] == ref[j : j + w]
        
    # Check if first window matches
    if h in ref_hashes:
        for j in ref_hashes[h]:
            if check_equal(0, j):
                out.append((0, j))
                
    # 4. Compute rolling hashes for subsequent windows in the submission text and match
    for i in range(w, len(text)):
        h = ((h - ord(text[i - w]) * powp) % mod + mod) % mod
        h = (h * base + ord(text[i])) % mod
        
        if h in ref_hashes:
            for j in ref_hashes[h]:
                if check_equal(i - w + 1, j):
                    out.append((i - w + 1, j))
                    
    return out
