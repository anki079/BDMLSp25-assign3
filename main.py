# main.py

import json
from rag_pipeline import RAGExperiment             # :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
from llama_rag_integration import LLaMARAGSystem    # :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}

def main():
    # ─── 1) Configuration ─────────────────────────────────
    TEXT_DIR      = "./processed_texts"
    CHUNK_SIZE    = 300
    CHUNK_OVERLAP = 50
    LLAMA_PATH    = "./checkpoints-llama-single-gpu-mem-opt"
    RESULTS_FILE  = "end_to_end_results.json"

    TEST_QUERIES = [
        "What are the main causes of climate change?",
        "How does global warming affect ocean levels?",
        "What are renewable energy solutions?"
    ]

    EMBEDDING_MODELS    = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-large-en"
    ]
    VECTOR_SEARCH_TYPES = [
        "faiss_flat",
        "faiss_ivf",
        "faiss_pq",
        "faiss_hnsw",
        "faiss_ivf_pq"
    ]

    all_results = {}

    # ─── 2) Sweep over embeddings & index types ────────────
    for em in EMBEDDING_MODELS:
        for vt in VECTOR_SEARCH_TYPES:
            tag = f"{em.split('/')[-1]}__{vt}"
            print(f"\n▶ Running experiment: {tag}")

            # 2a) Build & index
            exp = RAGExperiment(
                text_dir=TEXT_DIR,
                embedding_model=em,
                vector_search_type=vt,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            docs, load_t           = exp.load_documents()
            chunks, chunk_t        = exp.chunk_documents(docs)
            embeddings, embed_t    = exp.create_embeddings()
            vector_store, index_t  = exp.create_vector_store(chunks, embeddings)

            # 2b) Setup RAG + baseline
            rag_sys = LLaMARAGSystem(vector_store=vector_store,
                                     llama_path=LLAMA_PATH)
            llm     = rag_sys.setup_llama()

            # 2c) Evaluate RAG
            rag_per_query, rag_avg = rag_sys.evaluate_rag(
                llm, TEST_QUERIES, k=5
            )

            # 2d) Evaluate pure LLaMA baseline
            base_per_query, base_avg = rag_sys.evaluate_baseline(
                llm, TEST_QUERIES
            )

            # 2e) Store results for this experiment
            all_results[tag] = {
                "timings": {
                    "load_documents": load_t,
                    "chunk_documents": chunk_t,
                    "embedding_loading": embed_t,
                    "indexing": index_t
                },
                "rag": {
                    "per_query": rag_per_query,
                    "avg_total_time": rag_avg
                },
                "baseline": {
                    "per_query": base_per_query,
                    "avg_generation_time": base_avg
                }
            }

    # ─── 3) Write out everything ────────────────────────────
    with open(RESULTS_FILE, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nAll experiments complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
