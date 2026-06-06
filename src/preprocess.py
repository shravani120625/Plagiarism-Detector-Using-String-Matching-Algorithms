import re
import unicodedata
from typing import Tuple, List

def normalize(text: str) -> Tuple[str, List[int]]:
    """
    Normalizes the input text for robust string matching:
    - Normalizes unicode formatting (NFKC).
    - Converts all text to lowercase.
    - Filters out punctuation, symbols, and special characters, retaining only alphanumeric characters and spaces.
    - Collapses consecutive spaces into a single space.
    - Strips leading and trailing whitespace.
    
    Returns:
        - The normalized text string.
        - An index map: index `i` in the normalized string maps to the original character index in the raw text.
    """
    if not text:
        return "", []
        
    temp_chars = []
    temp_indices = []
    
    for idx, ch in enumerate(text):
        # Normalize unicode representations
        norm_ch = unicodedata.normalize("NFKC", ch)
        
        # Convert any whitespace character to a standard space
        if norm_ch.isspace():
            norm_ch = " "
            
        # Accumulate lowercase alphanumeric characters and standard spaces
        for c in norm_ch:
            if c.isalnum() or c == " ":
                temp_chars.append(c.lower())
                temp_indices.append(idx)
                
    # Collapse multiple consecutive spaces and strip leading spaces
    final_chars = []
    final_indices = []
    
    for i, c in enumerate(temp_chars):
        if c == " ":
            # Skip if it is a leading space or a consecutive space
            if len(final_chars) == 0 or final_chars[-1] == " ":
                continue
        final_chars.append(c)
        final_indices.append(temp_indices[i])
        
    # Strip trailing space if any exists
    if final_chars and final_chars[-1] == " ":
        final_chars.pop()
        final_indices.pop()
        
    return "".join(final_chars), final_indices


def split_into_sentences(text: str) -> List[Tuple[str, int]]:
    """
    Splits the raw text into sentences, returning each sentence along with its
    starting character offset in the original raw text.
    This helps in segment-based exact matching and granular highlighting.
    """
    # Simple regex to split sentences while capturing original position
    sentences = []
    # Find sentences using punctuation marks as boundaries
    pattern = re.compile(r'[^.!?]+[.!?]*')
    for match in pattern.finditer(text):
        s_text = match.group()
        s_start = match.start()
        # Clean sentence but keep it if it contains letters/numbers
        if any(c.isalnum() for c in s_text):
            sentences.append((s_text, s_start))
    return sentences
