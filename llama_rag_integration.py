# llama_rag_integration.py

import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

class LLaMARAGSystem:
    def __init__(self, vector_store, llama_path="./checkpoints-llama-single-gpu-mem-opt", max_new_tokens=512):
        self.vector_store = vector_store
        self.llama_path    = llama_path
        self.max_new_tokens = max_new_tokens

    def setup_llama(self):
        """Load LLaMA tokenizer + 4-bit model, wrapped in a LangChain LLM pipeline."""
        tokenizer = AutoTokenizer.from_pretrained(self.llama_path)
        model     = AutoModelForCausalLM.from_pretrained(
            self.llama_path,
            load_in_4bit=True,
            device_map="auto"
        )
        pipeline  = HuggingFacePipeline(
            model=model,
            tokenizer=tokenizer,
            device=0,
            max_new_tokens=self.max_new_tokens
        )
        return pipeline

    def evaluate_rag(self, llm, test_queries, k=5):
        """
        For each query:
          1) retrieve top-k docs
          2) build prompt = [chunks] + "Q: ... A:"
          3) generate answer
        Returns per-query retrieval, generation, total times + average total.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        results = []

        for q in test_queries:
            # 1) Retrieval
            t0 = time.time()
            docs = retriever.get_relevant_documents(q)
            t1 = time.time()

            # 2) Prompt construction
            context = "\n\n".join(d.page_content for d in docs)
            prompt  = f"{context}\n\nQ: {q}\nA:"

            # 3) Generation
            t2      = time.time()
            answer  = llm(prompt)
            t3      = time.time()

            results.append({
                "question": q,
                "answer": answer,
                "num_source_docs": len(docs),
                "retrieval_time": t1 - t0,
                "generation_time": t3 - t2,
                "total_time": t3 - t0
            })

        avg_total = sum(r["total_time"] for r in results) / len(results)
        return results, avg_total

    def evaluate_baseline(self, llm, test_queries):
        """
        Pure LLaMA run (no retrieval): 
        prompt = "Q: ... A:", measure only generation_time.
        """
        baseline = []

        for q in test_queries:
            prompt = f"Q: {q}\nA:"
            t0     = time.time()
            answer = llm(prompt)
            t1     = time.time()

            baseline.append({
                "question": q,
                "answer": answer,
                "generation_time": t1 - t0
            })

        avg_gen = sum(r["generation_time"] for r in baseline) / len(baseline)
        return baseline, avg_gen
