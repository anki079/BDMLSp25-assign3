import time
import json
import torch
import gc
import numpy as np
import os
import logging
import warnings
from datetime import datetime
from rag_pipeline import RAGExperiment
from llama_rag_integration import LLaMARAGSystem
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import matplotlib.pyplot as plt
from collections import defaultdict

def setup_logging(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"rag_experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    def warning_to_log(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{category.__name__}: {message}")
    
    warnings.showwarning = warning_to_log
    
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename

def main():
    log_file = setup_logging()
    
    # config
    TEXT_DIR      = "./processed_texts"
    CHUNK_SIZE    = 300
    CHUNK_OVERLAP = 50
    LLAMA_PATH    = "./checkpoints-llama-single-gpu-mem-opt"
    RESULTS_FILE  = "end_to_end_results.json"
    
    logging.info("Starting RAG system experiment")
    logging.info(f"Configuration: TEXT_DIR={TEXT_DIR}, CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")
    logging.info(f"LLAMA_PATH={LLAMA_PATH}, RESULTS_FILE={RESULTS_FILE}")

    TEST_QUERIES = [
    "What are the primary greenhouse gases contributing to climate change?",
    "How do rising sea levels impact coastal communities?",
    "What are the most promising renewable energy technologies for reducing carbon emissions?",
    "How does deforestation contribute to global warming?",
    "What are the economic impacts of transitioning to renewable energy sources?",
    "How do different countries compare in their climate change policies?",
    "What role does methane play in accelerating climate change?",
    "How can individuals reduce their carbon footprint?",
    "What are the challenges of implementing large-scale solar energy projects?",
    "How is climate change affecting agriculture and food security worldwide?"
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
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as fp:
                all_results = json.load(fp)
            logging.info(f"Loaded existing results from {RESULTS_FILE}")
            logging.info(f"Found {len(all_results)} previous experiment results")
        except json.JSONDecodeError:
            logging.warning(f"Error loading {RESULTS_FILE}, starting with empty results")
    
    logging.info("Setting up LLaMA model (one-time load)")
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
        logging.info("Tokenizer loaded successfully")
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH,
            device_map="auto"
        )
        logging.info(f"Model loaded successfully with device map: {model.hf_device_map}")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        logging.info(f"Hugging Face pipeline created")

        # wrapper for pipeline for batched processing
        class BatchCapableLLM:
            def __init__(self, pipeline):
                self.pipeline = pipeline
            
            def __call__(self, prompt):
                outputs = self.pipeline(prompt)
                return outputs[0]['generated_text'][len(prompt):]
            
            def batch(self, prompts):
                logging.info(f"Processing batch of {len(prompts)} prompts")
                outputs = self.pipeline(prompts)
                
                results = []
                for i, output in enumerate(outputs):
                    generated_text = output[0]['generated_text'][len(prompts[i]):]
                    results.append(generated_text)
                
                return results
        
        llm = BatchCapableLLM(pipe)
        model_load_time = time.time() - start_time
        logging.info(f"LLaMA model loaded successfully in {model_load_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f"Error loading LLaMA model: {e}", exc_info=True)
        raise

    for em in EMBEDDING_MODELS:
        logging.info(f"==========================================")
        logging.info(f"Processing embedding model: {em}")
        
        try:
            exp = RAGExperiment(
                text_dir=TEXT_DIR,
                embedding_model=em,
                vector_search_type="temp",
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            logging.info(f"Experiment initialized with embedding model: {em}")
            
            # once per embedding model
            docs, load_t = exp.load_documents()
            
            chunks, chunk_t = exp.chunk_documents(docs)
            
            embedding_model, embed_t = exp.create_embeddings()
            
            # one-time precomputing all embeddings for current model
            logging.info(f"Pre-computing embeddings for all chunks...")
            start_time = time.time()
            
            batch_size = 100
            batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
            all_chunk_embeddings = []
            
            for i, batch in enumerate(batches):
                batch_texts = [chunk.page_content for chunk in batch]
                batch_start = time.time()
                
                try:
                    if hasattr(embedding_model, 'embed_documents'):
                        batch_embeddings = embedding_model.embed_documents(batch_texts)
                        method = "batch"
                    else:
                        batch_embeddings = [embedding_model.embed_query(text) for text in batch_texts]
                        method = "individual"
                    
                    all_chunk_embeddings.extend(batch_embeddings)
                    batch_time = time.time() - batch_start
                    
                    if (i+1) % 5 == 0 or (i+1) == len(batches):
                        logging.info(f"Processed {min((i+1)*batch_size, len(chunks))}/{len(chunks)} chunks "
                                    f"using {method} embedding ({batch_time:.2f}s for this batch)")
                except Exception as e:
                    logging.error(f"Error in batch {i+1}: {str(e)}")
                    raise
            
            all_chunk_embeddings = np.vstack(all_chunk_embeddings).astype('float32')
            precompute_time = time.time() - start_time
            dim = all_chunk_embeddings.shape[1] if len(all_chunk_embeddings) > 0 else 0
            logging.info(f"Generated {len(all_chunk_embeddings)} embeddings (dim={dim}) in {precompute_time:.2f} seconds")
            
            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            
            logging.info("Evaluating baseline performance (LLM only, no retrieval)...")
            dummy_vectorstore = exp.create_dummy_vectorstore(embedding_model)
            rag_sys = LLaMARAGSystem(
                vector_store=dummy_vectorstore,
                llama_path=LLAMA_PATH
            )
            base_per_query, base_avg = rag_sys.evaluate_baseline_with_llm(
                llm, TEST_QUERIES
            )
            logging.info(f"Baseline evaluation complete. Average generation time: {base_avg:.2f} seconds")
            
            # testing diff index types with the same precomputed embeddings
            for vt in VECTOR_SEARCH_TYPES:
                tag = f"{em.split('/')[-1]}__{vt}"
                logging.info(f"----------------------------------------")
                logging.info(f"Testing index type: {tag}")
                
                if tag in all_results and "error" not in all_results[tag]:
                    logging.info(f"Experiment {tag} already completed, skipping")
                    continue
                
                try:
                    exp.vector_search_type = vt
                    
                    logging.info(f"Creating {vt} vector store...")
                    start_time = time.time()
                    vector_store = exp.create_vector_store_from_embeddings(
                        embedding_model,
                        texts,
                        metadatas,
                        all_chunk_embeddings
                    )
                    index_t = time.time() - start_time
                    logging.info(f"Vector store created in {index_t:.2f} seconds")
                    
                    logging.info(f"Setting up RAG system with {vt}...")
                    rag_sys = LLaMARAGSystem(
                        vector_store=vector_store,
                        llama_path=LLAMA_PATH
                    )
                    
                    logging.info(f"Evaluating RAG system with {vt}...")
                    rag_per_query, rag_avg = rag_sys.evaluate_rag_with_llm(
                        llm, TEST_QUERIES, k=5
                    )
                    logging.info(f"RAG evaluation complete. Average total time: {rag_avg:.2f} seconds")
                    
                    speedup = base_avg / rag_avg if rag_avg > 0 else 0
                    logging.info(f"Performance comparison for {tag}:")
                    logging.info(f"  - Baseline (LLM only): {base_avg:.2f}s")
                    logging.info(f"  - RAG with {vt}: {rag_avg:.2f}s")
                    logging.info(f"  - Speedup factor: {speedup:.2f}x")
                    
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
                    
                    # save results after each experiment
                    with open(RESULTS_FILE, "w") as fp:
                        json.dump(all_results, fp, indent=2)
                    logging.info(f"Results for {tag} saved to {RESULTS_FILE}")
                    
                    del vector_store
                    del rag_sys
                    gc.collect()
                    torch.cuda.empty_cache()
                    logging.info(f"Experiment {tag} completed successfully")
                    
                except Exception as e:
                    logging.error(f"Error in experiment {tag}: {str(e)}", exc_info=True)
                    all_results[tag] = {"error": str(e)}
                    with open(RESULTS_FILE, "w") as fp:
                        json.dump(all_results, fp, indent=2)
                    logging.info(f"Error for {tag} saved to {RESULTS_FILE}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
            
            logging.info("Cleaning up after completing all index types...")
            del embedding_model
            del chunks
            del docs
            del all_chunk_embeddings
            del exp
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("cleanup complete")
            
        except Exception as e:
            logging.error(f"Error with embedding model {em}: {str(e)}", exc_info=True)
            continue

    def analyze_inference_times(all_results):
        logging.info("Analyzing inference times across experiments...")
        
        retrieval_times = defaultdict(list)
        generation_times = defaultdict(list) 
        total_times = defaultdict(list)
        
        for config, result in all_results.items():
            if "error" in result:
                continue
                
            try:
                if "rag" in result and "per_query" in result["rag"]:
                    for query_result in result["rag"]["per_query"]:
                        retrieval_times[config].append(query_result["retrieval_time"])
                        generation_times[config].append(query_result["generation_time"])
                        total_times[config].append(query_result["total_time"])
            except Exception as e:
                logging.error(f"Error processing results for {config}: {str(e)}")
        
        stats = {}
        for config in total_times.keys():
            stats[config] = {
                "retrieval": {
                    "mean": np.mean(retrieval_times[config]),
                    "std": np.std(retrieval_times[config]),
                    "min": np.min(retrieval_times[config]),
                    "max": np.max(retrieval_times[config])
                },
                "generation": {
                    "mean": np.mean(generation_times[config]),
                    "std": np.std(generation_times[config]),
                    "min": np.min(generation_times[config]),
                    "max": np.max(generation_times[config])
                },
                "total": {
                    "mean": np.mean(total_times[config]),
                    "std": np.std(total_times[config]),
                    "min": np.min(total_times[config]),
                    "max": np.max(total_times[config]),
                    "n_samples": len(total_times[config])
                }
            }
        
        for config, stat in stats.items():
            logging.info(f"\nStatistics for {config}:")
            logging.info(f"  Total inference time: mean={stat['total']['mean']:.2f}s, std={stat['total']['std']:.2f}s")
            logging.info(f"  Retrieval time: mean={stat['retrieval']['mean']:.2f}s, std={stat['retrieval']['std']:.2f}s")
            logging.info(f"  Generation time: mean={stat['generation']['mean']:.2f}s, std={stat['generation']['std']:.2f}s")
            logging.info(f"  Samples: {stat['total']['n_samples']}")
        
        with open("inference_time_stats.json", "w") as fp:
            json.dump(stats, fp, indent=2)
        logging.info("Inference time statistics saved to inference_time_stats.json")
        
        return stats
    
    stats = analyze_inference_times(all_results)

    # def create_visualizations(stats):
    #     logging.info("Creating inference time visualizations...")
        
    #     configs = list(stats.keys())
    #     means = [stats[c]["total"]["mean"] for c in configs]
    #     stds = [stats[c]["total"]["std"] for c in configs]
    #     retrieval_means = [stats[c]["retrieval"]["mean"] for c in configs]
    #     generation_means = [stats[c]["generation"]["mean"] for c in configs]
        
    #     sorted_indices = np.argsort(means)
    #     configs = [configs[i] for i in sorted_indices]
    #     means = [means[i] for i in sorted_indices]
    #     stds = [stds[i] for i in sorted_indices]
    #     retrieval_means = [retrieval_means[i] for i in sorted_indices]
    #     generation_means = [generation_means[i] for i in sorted_indices]
        
    #     plt.figure(figsize=(12, 8))
        
    #     x = np.arange(len(configs))
    #     width = 0.35
        
    #     plt.bar(x, retrieval_means, width, label='Retrieval Time')
    #     plt.bar(x, generation_means, width, bottom=retrieval_means, label='Generation Time')
        
    #     plt.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', capsize=5)
        
    #     plt.xlabel('Configuration')
    #     plt.ylabel('Time (seconds)')
    #     plt.title('RAG Inference Time by Configuration')
    #     plt.xticks(x, configs, rotation=45, ha='right')
    #     plt.legend()
    #     plt.tight_layout()
        
    #     plt.savefig('inference_times.png', dpi=300)
    #     logging.info("Visualization saved to inference_times.png")

    # create_visualizations(stats)

    with open(RESULTS_FILE, "w") as fp:
        json.dump(all_results, fp, indent=2)
    logging.info(f"All experiments complete. Final results saved to {RESULTS_FILE}")

    experiment_count = len(all_results)
    successful_count = sum(1 for v in all_results.values() if "error" not in v)
    error_count = experiment_count - successful_count
    
    logging.info("======= EXPERIMENT SUMMARY =======")
    logging.info(f"Total experiments: {experiment_count}")
    logging.info(f"Successful: {successful_count}")
    logging.info(f"Failed: {error_count}")
    logging.info("=================================")

    logging.info("Performing final cleanup...")
    del model
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Final cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Unhandled exception in main: {str(e)}", exc_info=True)
        raise