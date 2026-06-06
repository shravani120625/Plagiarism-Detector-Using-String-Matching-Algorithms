import os
import argparse
import uvicorn
import json

# Import local pipeline modules
from src.preprocess import normalize
from src.exact import rabin_karp_windows
from src.winnow import winnow_fingerprints
from src.similarity import CustomTFIDF, cosine_similarity, ngram_set, jaccard
from src.scoring import calculate_exact_coverage, blended_score, map_normalized_spans_to_original, merge_spans

def run_cli_comparison(file_sub: str, file_ref: str, window_size: int = 40):
    """
    Performs a terminal-based comparison between a submission file and a reference file.
    Prints a detailed plagiarism scorecard in the CLI.
    """
    if not os.path.exists(file_sub):
        print(f"Error: Submission file '{file_sub}' does not exist.")
        return
    if not os.path.exists(file_ref):
        print(f"Error: Reference file '{file_ref}' does not exist.")
        return
        
    with open(file_sub, "r", encoding="utf-8") as f:
        sub_raw = f.read()
    with open(file_ref, "r", encoding="utf-8") as f:
        ref_raw = f.read()
        
    print("\n" + "="*50)
    print("      PLAGIARISM DETECTOR CLI SCORECARD")
    print("="*50)
    print(f"Submission: {os.path.basename(file_sub)} ({len(sub_raw)} chars)")
    print(f"Reference:  {os.path.basename(file_ref)} ({len(ref_raw)} chars)")
    print("-"*50)
    
    # 1. Preprocess
    sub_norm, sub_idx_map = normalize(sub_raw)
    ref_norm, ref_idx_map = normalize(ref_raw)
    
    # 2. Exact Matching (Rabin-Karp)
    rk_matches = rabin_karp_windows(sub_norm, ref_norm, w=window_size)
    exact_cov = calculate_exact_coverage(
        matches=[(m[0], window_size) for m in rk_matches],
        text_len=len(sub_norm)
    )
    
    # 3. Local Fingerprinting (Winnowing)
    sub_winnow = winnow_fingerprints(sub_norm, k=5, t=9)
    ref_winnow = winnow_fingerprints(ref_norm, k=5, t=9)
    sub_hashes = {h for _, h in sub_winnow}
    ref_hashes = {h for _, h in ref_winnow}
    shared_hashes = sub_hashes.intersection(ref_hashes)
    winnow_overlap = len(shared_hashes) / max(1, min(len(sub_hashes), len(ref_hashes)))
    
    # 4. Custom TF-IDF Cosine Similarity
    tfidf = CustomTFIDF()
    tfidf.fit([ref_norm, sub_norm])  # Mini corpus of the two files
    sub_tfidf = tfidf.transform(sub_norm)
    ref_tfidf = tfidf.transform(ref_norm)
    tfidf_cosine = cosine_similarity(sub_tfidf, ref_tfidf)
    
    # 5. Jaccard n-gram Similarity
    sub_ngrams = ngram_set(sub_norm, n=4)
    ref_ngrams = ngram_set(ref_norm, n=4)
    jaccard_val = jaccard(sub_ngrams, ref_ngrams)
    
    # 6. Blended Score
    final_score = blended_score(exact_cov, winnow_overlap, tfidf_cosine, jaccard_val)
    
    # 7. Highlight mappings (merged exact match spans)
    sub_norm_spans = [(m[0], window_size) for m in rk_matches]
    sub_raw_spans = map_normalized_spans_to_original(sub_norm_spans, sub_idx_map)
    merged_sub_spans = merge_spans(sub_raw_spans)
    
    print(f"Exact Matches (Rabin-Karp Coverage):  {exact_cov * 100:.1f}%")
    print(f"Fingerprint Overlap (Winnowing):     {winnow_overlap * 100:.1f}%")
    print(f"Vocabulary Cosine (TF-IDF):          {tfidf_cosine * 100:.1f}%")
    print(f"Local Jaccard Similarity (4-gram):    {jaccard_val * 100:.1f}%")
    print("-"*50)
    
    if final_score < 20:
        status_label = "CLEAN (Low Similarity)"
    elif final_score < 50:
        status_label = "MODERATE OVERLAP (Review Suggested)"
    else:
        status_label = "SEVERE MATCHING (Highly Plagiarized)"
        
    print(f"OVERALL SIMILARITY SCORE:            {final_score}%")
    print(f"STATUS:                              {status_label}")
    print("="*50)
    
    # Show sample matching fragments
    if merged_sub_spans:
        print("\n--- SAMPLE PLAGIARIZED EVIDENCE FRAGMENTS ---")
        for idx, (start, end) in enumerate(merged_sub_spans[:3]):
            # Print a snippet of matching text from the submission
            snippet = sub_raw[start:end+1].strip().replace('\n', ' ')
            if len(snippet) > 80:
                snippet = snippet[:80] + "..."
            print(f"{idx+1}. Offset {start}-{end}: \"{snippet}\"")
        if len(merged_sub_spans) > 3:
            print(f"... and {len(merged_sub_spans) - 3} more matching segments.")
        print("="*50)
        
    # Write JSON report to outputs/
    report_data = {
        "submission_file": file_sub,
        "reference_file": file_ref,
        "metrics": {
            "exact_coverage_pct": exact_cov * 100,
            "winnowing_overlap_pct": winnow_overlap * 100,
            "tfidf_cosine_pct": tfidf_cosine * 100,
            "jaccard_pct": jaccard_val * 100,
            "overall_similarity_pct": final_score
        },
        "evidence_count": len(merged_sub_spans),
        "evidence_spans": merged_sub_spans
    }
    
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/cli_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, indent=4)
    print(f"\nSaved detailed analysis report to: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plagiarism Detector Using String Matching Algorithms - CLI Tool"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # Run server command
    server_parser = subparsers.add_parser("run-server", help="Launch the web-based visual dashboard")
    server_parser.add_argument("--host", default="127.0.0.1", help="Server host IP")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port number")
    
    # Compare files command
    compare_parser = subparsers.add_parser("compare", help="Compare a submission against a reference file directly in CLI")
    compare_parser.add_argument("submission", help="Path to the submission file (.txt)")
    compare_parser.add_argument("reference", help="Path to the reference file (.txt)")
    compare_parser.add_argument("--window", type=int, default=40, help="Rabin-Karp character window size")
    
    args = parser.parse_args()
    
    if args.command == "run-server":
        print(f"Starting server on http://{args.host}:{args.port} ...")
        # Load uvicorn to run the FastAPI app
        uvicorn.run("src.app:app", host=args.host, port=args.port, reload=True)
    elif args.command == "compare":
        run_cli_comparison(args.submission, args.reference, args.window)
    else:
        # Default behavior: run server
        print("No command supplied. Defaulting to running the server...")
        print("Starting server on http://127.0.0.1:8000 ...")
        uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
