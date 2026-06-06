import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Tuple

# Import local modules
from src.preprocess import normalize, split_into_sentences
from src.exact import rabin_karp_windows
from src.winnow import winnow_fingerprints
from src.lsh import MinHasher, LSH
from src.similarity import CustomTFIDF, cosine_similarity, ngram_set, jaccard
from src.scoring import (
    map_normalized_spans_to_original,
    merge_spans,
    calculate_exact_coverage,
    blended_score
)

app = FastAPI(
    title="Plagiarism Detector API - Redesigned Dashboard Support",
    description="An elevated multi-stage plagiarism detection service supporting customizable parameters and CRUD operations on the reference index."
)

# Global database states
CORPUS: Dict[str, str] = {}           # doc_id -> raw text
NORM: Dict[str, str] = {}             # doc_id -> normalized text
IDX_MAPS: Dict[str, List[int]] = {}   # doc_id -> index map to raw text

# Initialize DSA modules
MH = MinHasher(n=100)
LSH_INDEX = LSH(bands=20, rows=5)
TFIDF_VEC = CustomTFIDF()

@app.on_event("startup")
def preload_corpus():
    """
    Pre-loads reference source documents from the 'documents/' folder on server startup.
    Only indexes files starting with 'source_'.
    """
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")
    if not os.path.exists(docs_dir):
        docs_dir = "documents"
        
    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.startswith("source_") and filename.endswith(".txt"):
                file_path = os.path.join(docs_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    norm_text, idx_map = normalize(text)
                    CORPUS[filename] = text
                    NORM[filename] = norm_text
                    IDX_MAPS[filename] = idx_map
                    
                    tokens = set(norm_text.split())
                    sig = MH.signature(tokens)
                    LSH_INDEX.add(filename, sig)
                    print(f"Pre-loaded reference document: {filename}")
                except Exception as e:
                    print(f"Error pre-loading {filename}: {e}")
                    
        # Fit TF-IDF Vectorizer
        if NORM:
            TFIDF_VEC.fit(list(NORM.values()))

# Request schemas
class DocSchema(BaseModel):
    doc_id: str
    text: str

class AnalyzeReqSchema(BaseModel):
    text: str
    top_k: int = 5
    window_size: int = 40
    w_exact: float = 0.4
    w_winnow: float = 0.3
    w_tfidf: float = 0.2
    w_jaccard: float = 0.1


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """
    Serves the elevated Single Page Application dashboard.
    """
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Frontend dashboard template not found.")
        
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/index")
def index_document(doc: DocSchema):
    """
    Normalizes a reference document, extracts signatures, and adds it to the index database.
    """
    doc_id = doc.doc_id.strip()
    text = doc.text.strip()
    
    if not doc_id or not text:
        raise HTTPException(status_code=400, detail="Document ID and text content cannot be empty.")
        
    # 1. Clean and normalize text, tracking original index mapping
    norm_text, idx_map = normalize(text)
    
    CORPUS[doc_id] = text
    NORM[doc_id] = norm_text
    IDX_MAPS[doc_id] = idx_map
    
    # 2. Extract MinHash signature and index in LSH tables
    tokens = set(norm_text.split())
    sig = MH.signature(tokens)
    LSH_INDEX.add(doc_id, sig)
    
    # 3. Fit/Re-fit the custom TF-IDF Vectorizer with all available reference documents
    TFIDF_VEC.fit(list(NORM.values()))
    
    return {"ok": True, "message": f"Successfully indexed '{doc_id}'."}


@app.get("/corpus")
def list_documents():
    """
    Lists all indexed documents and their statistics.
    """
    docs_summary = []
    for doc_id, raw_text in CORPUS.items():
        docs_summary.append({
            "doc_id": doc_id,
            "word_count": len(raw_text.split()),
            "char_count": len(raw_text)
        })
    return {"documents": docs_summary}


@app.delete("/corpus/{doc_id}")
def delete_document(doc_id: str):
    """
    Deletes a document from the reference index and re-indexes the LSH/TF-IDF models.
    """
    global TFIDF_VEC
    global LSH_INDEX
    
    doc_id = doc_id.strip()
    if doc_id not in CORPUS:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        
    # Remove from dicts
    del CORPUS[doc_id]
    del NORM[doc_id]
    del IDX_MAPS[doc_id]
    
    # Re-fit TF-IDF Vectorizer
    if NORM:
        TFIDF_VEC.fit(list(NORM.values()))
    else:
        TFIDF_VEC = CustomTFIDF()
        
    # Re-build LSH index
    LSH_INDEX = LSH(bands=20, rows=5)
    for d_id, norm_text in NORM.items():
        tokens = set(norm_text.split())
        sig = MH.signature(tokens)
        LSH_INDEX.add(d_id, sig)
        
    return {"ok": True, "message": f"Successfully deleted '{doc_id}' and re-indexed corpus."}


@app.post("/analyze")
def analyze_submission(req: AnalyzeReqSchema):
    """
    Runs multi-stage plagiarism detection comparing the submission text 
    against all candidate source files indexed in the corpus.
    """
    sub_raw = req.text.strip()
    if not sub_raw:
        raise HTTPException(status_code=400, detail="Submission text cannot be empty.")
        
    if not CORPUS:
        return {"overall": 0, "candidates": []}
        
    # 1. Normalize query/submission text
    sub_norm, sub_idx_map = normalize(sub_raw)
    sub_tokens = set(sub_norm.split())
    
    # 2. Candidate Selection via LSH (Retrieves matches in O(1) expected time)
    sub_sig = MH.signature(sub_tokens)
    candidates = LSH_INDEX.query(sub_sig)
    
    # Fallback: if LSH finds nothing but index contains documents, check all files
    if not candidates:
        candidates = list(CORPUS.keys())
        
    # 3. Perform detailed string matching and scoring on candidate list
    results = []
    sub_ngrams = ngram_set(sub_norm, n=4)
    sub_winnow_fp = winnow_fingerprints(sub_norm, k=5, t=9)
    sub_hashes = {h for _, h in sub_winnow_fp}
    
    # Pre-transform submission vector
    sub_tfidf = TFIDF_VEC.transform(sub_norm)
    
    for doc_id in candidates:
        ref_raw = CORPUS[doc_id]
        ref_norm = NORM[doc_id]
        ref_idx_map = IDX_MAPS[doc_id]
        
        # A. Exact Matching via Rabin-Karp Window Search
        window_size = req.window_size
        rk_matches = rabin_karp_windows(sub_norm, ref_norm, w=window_size)
        
        # B. Exact Coverage %
        exact_cov = calculate_exact_coverage(
            matches=[(m[0], window_size) for m in rk_matches],
            text_len=len(sub_norm)
        )
        
        # C. Local Fingerprinting Overlap (Winnowing)
        ref_winnow_fp = winnow_fingerprints(ref_norm, k=5, t=9)
        ref_hashes = {h for _, h in ref_winnow_fp}
        
        shared_hashes = sub_hashes.intersection(ref_hashes)
        winnow_overlap = len(shared_hashes) / max(1, min(len(sub_hashes), len(ref_hashes)))
        
        # D. Custom TF-IDF Cosine Similarity
        ref_tfidf = TFIDF_VEC.transform(ref_norm)
        tfidf_cosine = cosine_similarity(sub_tfidf, ref_tfidf)
        
        # E. Jaccard n-gram Similarity
        ref_ngrams = ngram_set(ref_norm, n=4)
        jaccard_val = jaccard(sub_ngrams, ref_ngrams)
        
        # F. Blended Score Calculation with custom weights
        score = blended_score(
            exact_cov, winnow_overlap, tfidf_cosine, jaccard_val,
            w_exact=req.w_exact,
            w_winnow=req.w_winnow,
            w_tfidf=req.w_tfidf,
            w_jaccard=req.w_jaccard
        )
        
        # G. Evidence Extraction & Highlight Alignment Spans
        sub_norm_spans = [(m[0], window_size) for m in rk_matches]
        ref_norm_spans = [(m[1], window_size) for m in rk_matches]
        
        # Map normalized spans back to original document indexes (Exact Matches - Red)
        sub_raw_spans = map_normalized_spans_to_original(sub_norm_spans, sub_idx_map)
        ref_raw_spans = map_normalized_spans_to_original(ref_norm_spans, ref_idx_map)
        
        sub_spans_merged = merge_spans(sub_raw_spans)
        ref_spans_merged = merge_spans(ref_raw_spans)
        
        # H. Paraphrase Sentence-level matching (Yellow Highlights)
        sub_sents = split_into_sentences(sub_raw)
        ref_sents = split_into_sentences(ref_raw)
        
        sub_para_spans = []
        ref_para_spans = []
        
        for s_sent, s_start in sub_sents:
            s_len = len(s_sent)
            s_norm_sent, _ = normalize(s_sent)
            if not s_norm_sent:
                continue
                
            # Calculate overlapping coverage with exact matches to avoid double-flagging
            sent_covered_len = 0
            for estart, eend in sub_spans_merged:
                overlap_start = max(s_start, estart)
                overlap_end = min(s_start + s_len - 1, eend)
                if overlap_start <= overlap_end:
                    sent_covered_len += (overlap_end - overlap_start + 1)
            
            # Skip if more than 40% of the sentence is already marked as exact copy-paste
            if s_len > 0 and (sent_covered_len / s_len) >= 0.4:
                continue
                
            # Perform word-level Jaccard similarity scan against reference sentences
            words_s = set(s_norm_sent.split())
            best_sim = 0.0
            best_ref_match = None
            
            for r_sent, r_start in ref_sents:
                r_norm_sent, _ = normalize(r_sent)
                if not r_norm_sent:
                    continue
                words_r = set(r_norm_sent.split())
                sim = jaccard(words_s, words_r)
                if sim > best_sim:
                    best_sim = sim
                    best_ref_match = (r_start, len(r_sent))
                    
            # 30% to 85% vocabulary overlap flags a paraphrase highlight (Yellow)
            if 0.30 <= best_sim < 0.85 and best_ref_match is not None:
                sub_para_spans.append((s_start, s_start + s_len - 1))
                ref_para_spans.append((best_ref_match[0], best_ref_match[0] + best_ref_match[1] - 1))
                
        # Merge overlapping paraphrase intervals
        sub_para_merged = merge_spans(sub_para_spans)
        ref_para_merged = merge_spans(ref_para_spans)
        
        results.append({
            "doc_id": doc_id,
            "score": score,
            "exact_coverage": exact_cov,
            "winnow_overlap": winnow_overlap,
            "tfidf_cosine": tfidf_cosine,
            "jaccard": jaccard_val,
            "submission_spans": sub_spans_merged,
            "reference_spans": ref_spans_merged,
            "submission_para_spans": sub_para_merged,
            "reference_para_spans": ref_para_merged,
            "raw_reference_text": ref_raw
        })
        
    # Sort results in descending order of similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = results[:req.top_k]
    
    # Overall score is the highest blended match in the dataset
    overall_score = top_candidates[0]["score"] if top_candidates else 0
    
    return {
        "overall": overall_score,
        "candidates": top_candidates
    }
