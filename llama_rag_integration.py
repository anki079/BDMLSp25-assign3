import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline


class LLaMARAGSystem:
    def __init__(self, vector_store, llama_path="./checkpoints-llama-single-gpu-mem-opt", max_new_tokens=512):
        self.vector_store = vector_store
        self.llama_path = llama_path
        self.max_new_tokens = max_new_tokens
        logging.info(f"LLaMARAGSystem initialized with max_new_tokens={max_new_tokens}")

    # load llaama tokenizer & 4-bit model wrapped in langchain llm pipeline
    def setup_llama(self):
        logging.info(f"Setting up LLaMA from {self.llama_path}")
        start_time = time.time()
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llama_path)
            logging.info("LLaMA tokenizer loaded")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.llama_path,
                load_in_4bit=True,
                device_map="auto"
            )
            logging.info(f"LLaMA model loaded with device map: {model.hf_device_map}")
            
            pipeline = HuggingFacePipeline(
                model=model,
                tokenizer=tokenizer,
                device=0,
                max_new_tokens=self.max_new_tokens
            )
            
            setup_time = time.time() - start_time
            logging.info(f"LLaMA setup completed in {setup_time:.2f} seconds")
            return pipeline
            
        except Exception as e:
            logging.error(f"Error setting up LLaMA: {str(e)}", exc_info=True)
            raise

    # methods to work with pre-loaded LLM
    def evaluate_rag_with_llm(self, llm, test_queries, k=5):
        logging.info(f"Evaluating RAG system with k={k} for {len(test_queries)} queries")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        results = []

        for i, q in enumerate(test_queries):
            logging.info(f"Processing query {i+1}/{len(test_queries)}: {q[:50]}...")

            t0 = time.time()
            docs = retriever.get_relevant_documents(q)
            t1 = time.time()
            retrieval_time = t1 - t0
            logging.info(f"  Retrieved {len(docs)} docs in {retrieval_time:.2f}s")

            context = "\n\n".join(d.page_content for d in docs)
            prompt = f"{context}\n\nQ: {q}\nA:"
            prompt_tokens = len(prompt.split())
            logging.info(f"  Prompt has ~{prompt_tokens} tokens")

            t2 = time.time()
            answer = llm(prompt)
            t3 = time.time()
            generation_time = t3 - t2
            logging.info(f"  Generated answer in {generation_time:.2f}s")

            total_time = t3 - t0
            results.append({
                "question": q,
                "answer": answer,
                "num_source_docs": len(docs),
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time
            })

        avg_total = sum(r["total_time"]      for r in results) / len(results)
        avg_retrieval = sum(r["retrieval_time"]  for r in results) / len(results)
        avg_generation = sum(r["generation_time"] for r in results) / len(results)

        logging.info("RAG Evaluation complete:")
        logging.info(f"  - Avg retrieval time:  {avg_retrieval:.2f}s")
        logging.info(f"  - Avg generation time: {avg_generation:.2f}s")
        logging.info(f"  - Avg total time:      {avg_total:.2f}s")

        return results, avg_total


    def evaluate_baseline_with_llm(self, llm, test_queries):
        logging.info(f"Evaluating baseline (LLM-only) for {len(test_queries)} queries using batch processing")
        
        prompts = [f"Q: {q}\nA:" for q in test_queries]
        
        start_time = time.time()
        batch_outputs = llm.batch(prompts)
        total_time = time.time() - start_time
        
        baseline = []
        for i, (query, output) in enumerate(zip(test_queries, batch_outputs)):
            baseline.append({
                "question": query,
                "answer": output,
                "generation_time": total_time / len(test_queries) #approx individual time
            })
            
        avg_gen = total_time / len(test_queries)
        logging.info(f"Baseline evaluation complete. Total time: {total_time:.2f}s, Average per query: {avg_gen:.2f}s")
        return baseline, avg_gen