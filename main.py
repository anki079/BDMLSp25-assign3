# main.py

import json
import torch
import gc
from rag_pipeline import RAGExperiment
from llama_rag_integration import LLaMARAGSystem
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

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

    # ─── 2) Load LLaMA model once ─────────────────────────
    print("\n▶ Setting up LLaMA model (one-time load)")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # Create the Hugging Face pipeline first
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Then create the LangChain wrapper around the pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✓ LLaMA model loaded successfully")

    # ─── 3) Sweep over embeddings & index types ────────────
    for em in EMBEDDING_MODELS:
        for vt in VECTOR_SEARCH_TYPES:
            tag = f"{em.split('/')[-1]}__{vt}"
            print(f"\n▶ Running experiment: {tag}")
            
            try:
                # 3a) Build & index
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

                # 3b) Setup RAG system but use pre-loaded LLM
                rag_sys = LLaMARAGSystem(
                    vector_store=vector_store,
                    llama_path=LLAMA_PATH
                )
                
                # 3c) Evaluate RAG with our pre-loaded LLM
                rag_per_query, rag_avg = rag_sys.evaluate_rag_with_llm(
                    llm, TEST_QUERIES, k=5
                )

                # 3d) Evaluate pure LLaMA baseline with pre-loaded LLM
                base_per_query, base_avg = rag_sys.evaluate_baseline_with_llm(
                    llm, TEST_QUERIES
                )

                # 3e) Store results for this experiment
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
                
                # Clean up to free memory after each experiment
                del vector_store
                del embeddings
                del exp
                gc.collect()
                torch.cuda.empty_cache()
                print(f"✓ Experiment {tag} completed successfully")
                
            except Exception as e:
                print(f"❌ Error in experiment {tag}: {str(e)}")
                all_results[tag] = {"error": str(e)}
                # Try to recover from error and continue
                gc.collect()
                torch.cuda.empty_cache()
                continue

    # ─── 4) Write out results and clean up ─────────────────
    with open(RESULTS_FILE, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n✓ All experiments complete. Results saved to {RESULTS_FILE}")

    # Final cleanup
    del model
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()