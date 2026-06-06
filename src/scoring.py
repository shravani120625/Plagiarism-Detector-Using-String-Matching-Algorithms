from typing import List, Tuple

def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Classic DSA Interval Merging (LeetCode 56: Merge Intervals).
    Merges overlapping or adjacent character intervals to produce clean 
    and concise spans for highlighting.
    
    Parameters:
        spans: List of tuples representing (start_char_index, end_char_index) inclusive.
        
    Returns:
        A list of merged non-overlapping intervals, sorted by start position.
    """
    if not spans:
        return []
        
    # 1. Sort intervals by start index
    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]
    
    # 2. Iterate and merge overlapping intervals
    for current in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        
        # If current interval overlaps or is adjacent to the previous one
        if curr_start <= prev_end + 1:
            # Update the end of the previous interval if the current one extends further
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
            
    return merged


def map_normalized_spans_to_original(
    spans: List[Tuple[int, int]], 
    idx_map: List[int]
) -> List[Tuple[int, int]]:
    """
    Maps matching spans in the normalized string back to original raw text character offsets
    using the precomputed index map.
    
    Parameters:
        spans: List of tuples representing (normalized_start_index, length)
        idx_map: List where idx_map[norm_index] = raw_index
        
    Returns:
        List of raw character spans: (raw_start, raw_end) inclusive.
    """
    if not idx_map or not spans:
        return []
        
    raw_spans = []
    max_norm_idx = len(idx_map) - 1
    
    for norm_start, length in spans:
        if norm_start > max_norm_idx:
            continue
            
        # Get start index in raw text
        raw_start = idx_map[norm_start]
        
        # Determine ending normalized character
        norm_end = min(norm_start + length - 1, max_norm_idx)
        raw_end = idx_map[norm_end]
        
        raw_spans.append((raw_start, raw_end))
        
    return raw_spans


def calculate_exact_coverage(matches: List[Tuple[int, int]], text_len: int) -> float:
    """
    Calculates the percentage of the normalized text that is covered by exact matches.
    Uses an array-based coverage tracker.
    
    Parameters:
        matches: List of tuples (start_idx, length) in normalized text.
        text_len: Length of the normalized text.
    """
    if text_len == 0 or not matches:
        return 0.0
        
    coverage = [0] * text_len
    for start, length in matches:
        # Fill matched intervals
        for i in range(start, min(start + length, text_len)):
            coverage[i] = 1
            
    return sum(coverage) / text_len


def blended_score(
    exact_cov: float, 
    winnow_overlap: float, 
    tfidf_cosine: float, 
    jaccard_similarity: float,
    w_exact: float = 0.4,
    w_winnow: float = 0.3,
    w_tfidf: float = 0.2,
    w_jaccard: float = 0.1
) -> int:
    """
    Combines different plagiarism metric indicators into a single unified 0-100 score
    using customizable weights.
    """
    # Normalize weights to sum to 1.0 if they don't already
    total_w = w_exact + w_winnow + w_tfidf + w_jaccard
    if total_w > 0:
        w_exact /= total_w
        w_winnow /= total_w
        w_tfidf /= total_w
        w_jaccard /= total_w
    else:
        w_exact, w_winnow, w_tfidf, w_jaccard = 0.4, 0.3, 0.2, 0.1
    
    raw_score = (
        w_exact * exact_cov +
        w_winnow * winnow_overlap +
        w_tfidf * tfidf_cosine +
        w_jaccard * jaccard_similarity
    )
    
    # Scale to 0-100 percentage
    return int(min(100, max(0, round(raw_score * 100))))
