# rag_pipeline.py
import os
import time
import json
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# custom wrapper for BGE embeddings to work with langchain
class CustomBGEEmbeddings:
    # from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    # model_name = "BAAI/bge-large-en-v1.5"
    # model_kwargs = {'device': 'cuda'}
    # encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    # model = HuggingFaceBgeEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs,
    #     query_instruction="为这个句子生成表示以用于检索相关文章："
    # )
    # model.query_instruction = "为这个句子生成表示以用于检索相关文章："
    
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                embeddings.append(embedding[0])
        
        return embeddings
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]


class RAGExperiment:
    def __init__(self, 
                 text_dir="./processed_texts",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 vector_search_type="faiss_flat",
                 chunk_size=300,
                 chunk_overlap=50,
                 results_dir="./experiment_results",
                 openai_api_key=None):
        
        self.text_dir = text_dir
        self.embedding_model = embedding_model
        self.vector_search_type = vector_search_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.results_dir = results_dir
        self.openai_api_key = openai_api_key
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Experiment ID based on configuration
        self.experiment_id = f"{embedding_model.split('/')[-1]}_{vector_search_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        

    def load_documents(self):
        start_time = time.time()
        text_loader = DirectoryLoader(self.text_dir, glob="*.txt", loader_cls=TextLoader)
        documents = text_loader.load()
        metadata_dir = os.path.join(self.text_dir, "metadata")
        for doc in documents:
            txt_filename = os.path.basename(doc.metadata['source'])
            json_filename = txt_filename.replace('.txt', '_metadata.json')
            metadata_path = os.path.join(metadata_dir, json_filename)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    doc.metadata.update(metadata)
        
        load_time = time.time() - start_time
        print(f"Loaded {len(documents)} documents with metadata in {load_time:.2f} seconds")
        return documents, load_time
    
    def chunk_documents(self, documents):
        start_time = time.time()
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
            # separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_documents([doc])
            for chunk in doc_chunks:
                chunk.metadata.update({
                    "source_document": doc.metadata.get('filename', ''),
                    "total_pages": doc.metadata.get('total_pages', 0)
                })
            chunks.extend(doc_chunks)
        
        chunk_time = time.time() - start_time
        print(f"Created {len(chunks)} chunks in {chunk_time:.2f} seconds")
        return chunks, chunk_time
    
    def create_embeddings(self):
        start_time = time.time()
        
        # if "openai" in self.embedding_model.lower():
        #     if not self.openai_api_key:
        #         raise ValueError("API key required for OpenAI embeddings")
        #     embeddings = OpenAIEmbeddings(
        #         model="text-embedding-3-large",
        #         openai_api_key=self.openai_api_key
        #     )
        if "bge-large" in self.embedding_model.lower():
            embeddings = CustomBGEEmbeddings(model_name="BAAI/bge-large-en")
        else:
            # default to sentence-transformers
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
        
        embedding_time = time.time() - start_time
        print(f"Loaded embedding model in {embedding_time:.2f} seconds")
        return embeddings, embedding_time
    
    def create_faiss_index(self, dimension):
        if self.vector_search_type == "faiss_flat":
            index = faiss.IndexFlatL2(dimension)
        
        elif self.vector_search_type == "faiss_ivf":
            nlist = 100  # no of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        elif self.vector_search_type == "faiss_pq":
            m = 8  # no of subquantizers
            nbits = 8  # bits per subquantizer
            index = faiss.IndexPQ(dimension, m, nbits)
        
        elif self.vector_search_type == "faiss_hnsw":
            M = 32  # no of connections per element
            index = faiss.IndexHNSWFlat(dimension, M)
        
        elif self.vector_search_type == "faiss_ivf_pq":
            nlist = 100
            m = 8
            nbits = 8
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        
        else:
            raise ValueError(f"Unsupported vector search type: {self.vector_search_type}")
        print(f"Created faiss index of type {self.vector_search_type} with dimension {dimension}")
        return index
    
    def create_vector_store(self, chunks, embeddings):
        """Create a LangChain FAISS vector store using a manually built FAISS index."""

        start_time = time.time()

        # 1) Determine embedding dimensionality
        sample_emb = embeddings.embed_query("test")
        dimension = len(sample_emb)

        # 2) Build the appropriate FAISS index (flat, ivf, pq, hnsw, ivf_pq)
        faiss_index = self.create_faiss_index(dimension)  # Renamed to faiss_index to avoid conflict

        # 3) Embed every chunk into a single matrix
        all_embeddings = np.vstack([
            embeddings.embed_query(chunk.page_content)
            for chunk in chunks
        ]).astype('float32') 

        # 4) Train the index if it requires training (e.g. IVF, PQ)
        if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
            faiss_index.train(all_embeddings)

        # 5) Add all vectors into the index
        faiss_index.add(all_embeddings)

        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata     for c in chunks]

        # 6) Create FAISS vectorstore WITHOUT passing the index parameter
        #    and then manually set the index afterward
        vectorstore = FAISS.from_texts(
            texts,
            embeddings,
            metadatas=metadatas
        )
        
        # Replace the default index with our custom one
        vectorstore.index = faiss_index

        # 7) (Optional) tweak index parameters per type
        if "ivf" in self.vector_search_type.lower():
            # make sure IVF also uses a reasonable nprobe
            if hasattr(vectorstore.index, "nprobe"):
                vectorstore.index.nprobe = 10
        if "hnsw" in self.vector_search_type.lower():
            # how exhaustive the search is
            if hasattr(vectorstore.index, "hnsw"):
                vectorstore.index.hnsw.efConstruction = 200
                vectorstore.index.hnsw.efSearch = 200

        index_time = time.time() - start_time
        print(f"Created vector store in {index_time:.2f} seconds")

        return vectorstore, index_time

    def create_dummy_vectorstore(self, embedding_model):
        """Create a simple FAISS vectorstore just for baseline testing."""
        return FAISS.from_texts(
            ["dummy text"],
            embedding_model
        )

    def create_vector_store_from_embeddings(self, embedding_model, texts, metadatas, precomputed_embeddings):
        """Create a vector store using precomputed embeddings."""
        
        # 1) Determine embedding dimensionality
        dimension = precomputed_embeddings.shape[1]
        
        # 2) Build the appropriate FAISS index
        faiss_index = self.create_faiss_index(dimension)
        
        # 3) Train the index if it requires training (e.g. IVF, PQ)
        if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
            print(f"Training FAISS index of type {self.vector_search_type}...")
            faiss_index.train(precomputed_embeddings)
        
        # 4) Add all vectors into the index
        faiss_index.add(precomputed_embeddings)
        print(f"Added {len(precomputed_embeddings)} vectors to the FAISS index")
        
        # 5) Create FAISS vectorstore WITHOUT passing the custom index parameter
        vectorstore = FAISS.from_texts(
            texts,
            embedding_model,
            metadatas=metadatas
        )
        
        # 6) Replace the default index with our custom one
        vectorstore.index = faiss_index
        
        # 7) Tweak index parameters per type
        if "ivf" in self.vector_search_type.lower():
            if hasattr(vectorstore.index, "nprobe"):
                vectorstore.index.nprobe = 10
        if "hnsw" in self.vector_search_type.lower():
            if hasattr(vectorstore.index, "hnsw"):
                vectorstore.index.hnsw.efConstruction = 200
                vectorstore.index.hnsw.efSearch = 200
                
        return vectorstore
    
    def run_experiment(self, test_queries=None):
        """Run complete experiment and save results."""
        results = {
            "experiment_id": self.experiment_id,
            "embedding_model": self.embedding_model,
            "vector_search_type": self.vector_search_type,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "timings": {}
        }
        
        # Load documents
        documents, load_time = self.load_documents()
        results["timings"]["document_loading"] = load_time
        
        # Chunk documents
        chunks, chunk_time = self.chunk_documents(documents)
        results["timings"]["chunking"] = chunk_time
        results["num_chunks"] = len(chunks)
        
        # Create embeddings
        embeddings, embedding_time = self.create_embeddings()
        results["timings"]["embedding_loading"] = embedding_time
        
        # Create vector store
        vectorstore, index_time = self.create_vector_store(chunks, embeddings)
        results["timings"]["indexing"] = index_time
        
        # Test retrieval if queries provided
        if test_queries:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            query_results = []
            
            total_query_time = 0
            for query in test_queries:
                start_time = time.time()
                docs = retriever.get_relevant_documents(query)
                query_time = time.time() - start_time
                total_query_time += query_time
                
                query_results.append({
                    "query": query,
                    "time": query_time,
                    "num_docs": len(docs)
                })
            
            results["query_results"] = query_results
            results["timings"]["avg_query_time"] = total_query_time / len(test_queries)
        
        # Save results
        results_path = os.path.join(self.results_dir, f"{self.experiment_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        return results


# Example usage
if __name__ == "__main__":
    # Example test queries
    test_queries = [
        "What are the main causes of climate change?",
        "How does global warming affect ocean levels?",
        "What are renewable energy solutions?"
    ]
    
    # List of experiments to run
    experiments = []
    
    # Define the embedding models and vector search types
    embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-large-en"
        # "openai/text-embedding-3-large"
    ]
    
    vector_search_types = [
        "faiss_flat",
        "faiss_ivf",
        "faiss_pq",
        "faiss_hnsw",
        "faiss_ivf_pq"
    ]
    
    # Create all combinations
    for embedding_model in embedding_models:
        for vector_search_type in vector_search_types:
            config = {
                "embedding_model": embedding_model,
                "vector_search_type": vector_search_type
            }
            # Add OpenAI API key if needed
            if "openai" in embedding_model.lower():
                config["openai_api_key"] = "your-openai-api-key-here"
            
            experiments.append(config)
    
    # Run experiments
    for config in experiments:
        try:
            exp = RAGExperiment(**config)
            exp.run_experiment(test_queries)
        except Exception as e:
            print(f"Experiment failed for {config}: {str(e)}")
            continue