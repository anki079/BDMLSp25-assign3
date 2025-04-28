# main.py
import time
import json
import torch
import gc
import numpy as np
from rag_pipeline import RAGExperiment
from llama_rag_integration import LLaMARAGSystem
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
    
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH,
        device_map="auto"
    )

    # Create the Hugging Face pipeline first
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    # Then create the LangChain wrapper around the pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✓ LLaMA model loaded successfully")

    # Optimize: process one embedding model at a time and precompute embeddings
    for em in EMBEDDING_MODELS:
        print(f"\n▶ Processing embedding model: {em}")
        
        try:
            # Setup experiment with the current embedding model
            exp = RAGExperiment(
                text_dir=TEXT_DIR,
                embedding_model=em,
                vector_search_type="temp",  # Temporary value, will be updated later
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            # These steps only need to happen once per embedding model
            docs, load_t = exp.load_documents()
            chunks, chunk_t = exp.chunk_documents(docs)
            embedding_model, embed_t = exp.create_embeddings()
            
            # Pre-compute all embeddings for this model just once
            print(f"Pre-computing embeddings for all chunks...")
            start_time = time.time()
            
            # Process in batches to avoid memory issues
            batch_size = 100
            batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
            all_chunk_embeddings = []
            
            for i, batch in enumerate(batches):
                batch_texts = [chunk.page_content for chunk in batch]
                if hasattr(embedding_model, 'embed_documents'):
                    # If the embedding model supports batch processing
                    batch_embeddings = embedding_model.embed_documents(batch_texts)
                else:
                    # Fallback to individual processing
                    batch_embeddings = [embedding_model.embed_query(text) for text in batch_texts]
                
                all_chunk_embeddings.extend(batch_embeddings)
                
                if (i+1) % 10 == 0 or (i+1) == len(batches):
                    print(f"Processed {(i+1)*batch_size}/{len(chunks)} chunks")
            
            all_chunk_embeddings = np.vstack(all_chunk_embeddings).astype('float32')
            precompute_time = time.time() - start_time
            print(f"✓ Generated {len(all_chunk_embeddings)} embeddings in {precompute_time:.2f} seconds")
            
            # Store the texts and metadata separately for FAISS creation
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            
            # Evaluate baseline once per embedding model
            dummy_vectorstore = exp.create_dummy_vectorstore(embedding_model)
            rag_sys = LLaMARAGSystem(
                vector_store=dummy_vectorstore,
                llama_path=LLAMA_PATH
            )
            base_per_query, base_avg = rag_sys.evaluate_baseline_with_llm(
                llm, TEST_QUERIES
            )
            
            # Now test different index types with the same precomputed embeddings
            for vt in VECTOR_SEARCH_TYPES:
                tag = f"{em.split('/')[-1]}__{vt}"
                print(f"\n▶ Testing index type: {tag}")
                
                try:
                    # Update vector search type
                    exp.vector_search_type = vt
                    
                    # Create vector store with precomputed embeddings
                    start_time = time.time()
                    vector_store = exp.create_vector_store_from_embeddings(
                        embedding_model,
                        texts,
                        metadatas,
                        all_chunk_embeddings
                    )
                    index_t = time.time() - start_time
                    
                    # Setup RAG system and evaluate
                    rag_sys = LLaMARAGSystem(
                        vector_store=vector_store,
                        llama_path=LLAMA_PATH
                    )
                    
                    # Evaluate RAG with pre-loaded LLM
                    rag_per_query, rag_avg = rag_sys.evaluate_rag_with_llm(
                        llm, TEST_QUERIES, k=5
                    )
                    
                    # Store results
                    all_results[tag] = {
                        "timings": {
                            "load_documents": load_t,
                            "chunk_documents": chunk_t,
                            "embedding_loading": embed_t,
                            "embedding_generation": precompute_time,
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
                    
                    # Clean up vector store to free memory
                    del vector_store
                    del rag_sys
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"✓ Experiment {tag} completed successfully")
                    
                except Exception as e:
                    print(f"❌ Error in experiment {tag}: {str(e)}")
                    all_results[tag] = {"error": str(e)}
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
            
            # Clean up after processing all index types
            del embedding_model
            del chunks
            del docs
            del all_chunk_embeddings
            del exp
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error with embedding model {em}: {str(e)}")
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