# # rag_pipeline.py
# import os
# import time
# import json
# from datetime import datetime
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
# from langchain_community.vectorstores import FAISS
# import faiss
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModel

# # custom wrapper for BGE embeddings to work with langchain
# # class CustomBGEEmbeddings:
# #     # from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# #     # model_name = "BAAI/bge-large-en-v1.5"
# #     # model_kwargs = {'device': 'cuda'}
# #     # encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# #     # model = HuggingFaceBgeEmbeddings(
# #     #     model_name=model_name,
# #     #     model_kwargs=model_kwargs,
# #     #     encode_kwargs=encode_kwargs,
# #     #     query_instruction="为这个句子生成表示以用于检索相关文章："
# #     # )
# #     # model.query_instruction = "为这个句子生成表示以用于检索相关文章："
    
# #     def __init__(self, model_name="BAAI/bge-large-en"):
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
# #         self.model = AutoModel.from_pretrained(model_name)
# #         self.model.eval()
# #         if torch.cuda.is_available():
# #             self.model.to('cuda')
    
# #     def embed_documents(self, texts):
# #         embeddings = []
# #         for text in texts:
# #             inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
# #             if torch.cuda.is_available():
# #                 inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
# #             with torch.no_grad():
# #                 outputs = self.model(**inputs)
# #                 embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
# #                 # Normalize the embedding
# #                 embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
# #                 embeddings.append(embedding[0])
        
# #         return embeddings
    
# #     def embed_query(self, text):
# #         return self.embed_documents([text])[0]


# class RAGExperiment:
#     def __init__(self, 
#                  text_dir="./processed_texts",
#                  embedding_model="sentence-transformers/all-MiniLM-L6-v2",
#                  vector_search_type="faiss_flat",
#                  chunk_size=300,
#                  chunk_overlap=50,
#                  results_dir="./experiment_results",
#                  openai_api_key=None):
        
#         self.text_dir = text_dir
#         self.embedding_model = embedding_model
#         self.vector_search_type = vector_search_type
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.results_dir = results_dir
#         self.openai_api_key = openai_api_key
        
#         # Create results directory
#         os.makedirs(results_dir, exist_ok=True)
        
#         # Experiment ID based on configuration
#         self.experiment_id = f"{embedding_model.split('/')[-1]}_{vector_search_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        

#     def load_documents(self):
#         start_time = time.time()
#         text_loader = DirectoryLoader(self.text_dir, glob="*.txt", loader_cls=TextLoader)
#         documents = text_loader.load()
#         metadata_dir = os.path.join(self.text_dir, "metadata")
#         for doc in documents:
#             txt_filename = os.path.basename(doc.metadata['source'])
#             json_filename = txt_filename.replace('.txt', '_metadata.json')
#             metadata_path = os.path.join(metadata_dir, json_filename)
#             if os.path.exists(metadata_path):
#                 with open(metadata_path, 'r') as f:
#                     metadata = json.load(f)
#                     doc.metadata.update(metadata)
        
#         load_time = time.time() - start_time
#         print(f"Loaded {len(documents)} documents with metadata in {load_time:.2f} seconds")
#         return documents, load_time
    
#     def chunk_documents(self, documents):
#         start_time = time.time()
#         text_splitter = TokenTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap
#             # separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         chunks = []
#         for doc in documents:
#             doc_chunks = text_splitter.split_documents([doc])
#             for chunk in doc_chunks:
#                 chunk.metadata.update({
#                     "source_document": doc.metadata.get('filename', ''),
#                     "total_pages": doc.metadata.get('total_pages', 0)
#                 })
#             chunks.extend(doc_chunks)
        
#         chunk_time = time.time() - start_time
#         print(f"Created {len(chunks)} chunks in {chunk_time:.2f} seconds")
#         return chunks, chunk_time
    
#     def create_embeddings(self):
#         start_time = time.time()
        
#         # if "openai" in self.embedding_model.lower():
#         #     if not self.openai_api_key:
#         #         raise ValueError("API key required for OpenAI embeddings")
#         #     embeddings = OpenAIEmbeddings(
#         #         model="text-embedding-3-large",
#         #         openai_api_key=self.openai_api_key
#         #     )
#         if "bge-large" in self.embedding_model.lower():
#             from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        
#             embeddings = HuggingFaceBgeEmbeddings(
#                 model_name="BAAI/bge-large-en",
#                 model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
#                 encode_kwargs={'normalize_embeddings': True}
#             )
#         else:
#             # default to sentence-transformers
#             embeddings = HuggingFaceEmbeddings(
#                 model_name=self.embedding_model,
#                 model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
#             )
        
#         embedding_time = time.time() - start_time
#         print(f"Loaded embedding model in {embedding_time:.2f} seconds")
#         return embeddings, embedding_time
    
#     def create_faiss_index(self, dimension):
#         if self.vector_search_type == "faiss_flat":
#             index = faiss.IndexFlatL2(dimension)
        
#         elif self.vector_search_type == "faiss_ivf":
#             nlist = 100  # no of clusters
#             quantizer = faiss.IndexFlatL2(dimension)
#             index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
#         elif self.vector_search_type == "faiss_pq":
#             m = 8  # no of subquantizers
#             nbits = 8  # bits per subquantizer
#             index = faiss.IndexPQ(dimension, m, nbits)
        
#         elif self.vector_search_type == "faiss_hnsw":
#             M = 32  # no of connections per element
#             index = faiss.IndexHNSWFlat(dimension, M)
        
#         elif self.vector_search_type == "faiss_ivf_pq":
#             nlist = 100
#             m = 8
#             nbits = 8
#             quantizer = faiss.IndexFlatL2(dimension)
#             index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
        
#         else:
#             raise ValueError(f"Unsupported vector search type: {self.vector_search_type}")
#         print(f"Created faiss index of type {self.vector_search_type} with dimension {dimension}")
#         return index
    
#     def create_vector_store(self, chunks, embeddings):
#         """Create a LangChain FAISS vector store using a manually built FAISS index."""

#         start_time = time.time()

#         # 1) Determine embedding dimensionality
#         sample_emb = embeddings.embed_query("test")
#         dimension = len(sample_emb)

#         # 2) Build the appropriate FAISS index (flat, ivf, pq, hnsw, ivf_pq)
#         faiss_index = self.create_faiss_index(dimension)  # Renamed to faiss_index to avoid conflict

#         # 3) Embed every chunk into a single matrix
#         all_embeddings = np.vstack([
#             embeddings.embed_query(chunk.page_content)
#             for chunk in chunks
#         ]).astype('float32') 

#         # 4) Train the index if it requires training (e.g. IVF, PQ)
#         if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
#             faiss_index.train(all_embeddings)

#         # 5) Add all vectors into the index
#         faiss_index.add(all_embeddings)

#         texts     = [c.page_content for c in chunks]
#         metadatas = [c.metadata     for c in chunks]

#         # 6) Create FAISS vectorstore WITHOUT passing the index parameter
#         #    and then manually set the index afterward
#         vectorstore = FAISS.from_texts(
#             texts,
#             embeddings,
#             metadatas=metadatas
#         )
        
#         # Replace the default index with our custom one
#         vectorstore.index = faiss_index

#         # 7) (Optional) tweak index parameters per type
#         if "ivf" in self.vector_search_type.lower():
#             # make sure IVF also uses a reasonable nprobe
#             if hasattr(vectorstore.index, "nprobe"):
#                 vectorstore.index.nprobe = 10
#         if "hnsw" in self.vector_search_type.lower():
#             # how exhaustive the search is
#             if hasattr(vectorstore.index, "hnsw"):
#                 vectorstore.index.hnsw.efConstruction = 200
#                 vectorstore.index.hnsw.efSearch = 200

#         index_time = time.time() - start_time
#         print(f"Created vector store in {index_time:.2f} seconds")

#         return vectorstore, index_time

#     def create_dummy_vectorstore(self, embedding_model):
#         """Create a simple FAISS vectorstore just for baseline testing."""
#         return FAISS.from_texts(
#             ["dummy text"],
#             embedding_model
#         )

#     def create_vector_store_from_embeddings(self, embedding_model, texts, metadatas, precomputed_embeddings):
#         """Create a vector store using precomputed embeddings."""
        
#         # 1) Determine embedding dimensionality
#         dimension = precomputed_embeddings.shape[1]
        
#         # 2) Build the appropriate FAISS index
#         faiss_index = self.create_faiss_index(dimension)
        
#         # 3) Train the index if it requires training (e.g. IVF, PQ)
#         if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
#             print(f"Training FAISS index of type {self.vector_search_type}...")
#             faiss_index.train(precomputed_embeddings)
        
#         # 4) Add all vectors into the index
#         faiss_index.add(precomputed_embeddings)
#         print(f"Added {len(precomputed_embeddings)} vectors to the FAISS index")
        
#         # 5) Create FAISS vectorstore WITHOUT passing the custom index parameter
#         vectorstore = FAISS.from_texts(
#             texts,
#             embedding_model,
#             metadatas=metadatas
#         )
        
#         # 6) Replace the default index with our custom one
#         vectorstore.index = faiss_index
        
#         # 7) Tweak index parameters per type
#         if "ivf" in self.vector_search_type.lower():
#             if hasattr(vectorstore.index, "nprobe"):
#                 vectorstore.index.nprobe = 10
#         if "hnsw" in self.vector_search_type.lower():
#             if hasattr(vectorstore.index, "hnsw"):
#                 vectorstore.index.hnsw.efConstruction = 200
#                 vectorstore.index.hnsw.efSearch = 200
                
#         return vectorstore
    
#     # def run_experiment(self, test_queries=None):
#     #     """Run complete experiment and save results."""
#     #     results = {
#     #         "experiment_id": self.experiment_id,
#     #         "embedding_model": self.embedding_model,
#     #         "vector_search_type": self.vector_search_type,
#     #         "chunk_size": self.chunk_size,
#     #         "chunk_overlap": self.chunk_overlap,
#     #         "timings": {}
#     #     }
        
#     #     # Load documents
#     #     documents, load_time = self.load_documents()
#     #     results["timings"]["document_loading"] = load_time
        
#     #     # Chunk documents
#     #     chunks, chunk_time = self.chunk_documents(documents)
#     #     results["timings"]["chunking"] = chunk_time
#     #     results["num_chunks"] = len(chunks)
        
#     #     # Create embeddings
#     #     embeddings, embedding_time = self.create_embeddings()
#     #     results["timings"]["embedding_loading"] = embedding_time
        
#     #     # Create vector store
#     #     vectorstore, index_time = self.create_vector_store(chunks, embeddings)
#     #     results["timings"]["indexing"] = index_time
        
#     #     # Test retrieval if queries provided
#     #     if test_queries:
#     #         retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     #         query_results = []
            
#     #         total_query_time = 0
#     #         for query in test_queries:
#     #             start_time = time.time()
#     #             docs = retriever.get_relevant_documents(query)
#     #             query_time = time.time() - start_time
#     #             total_query_time += query_time
                
#     #             query_results.append({
#     #                 "query": query,
#     #                 "time": query_time,
#     #                 "num_docs": len(docs)
#     #             })
            
#     #         results["query_results"] = query_results
#     #         results["timings"]["avg_query_time"] = total_query_time / len(test_queries)
        
#     #     # Save results
#     #     results_path = os.path.join(self.results_dir, f"{self.experiment_id}_results.json")
#     #     with open(results_path, 'w') as f:
#     #         json.dump(results, f, indent=2)
        
#     #     print(f"Results saved to {results_path}")
#     #     return results


# # Example usage
# if __name__ == "__main__":
#     # Example test queries
#     test_queries = [
#         "What are the main causes of climate change?",
#         "How does global warming affect ocean levels?",
#         "What are renewable energy solutions?"
#     ]
    
#     # List of experiments to run
#     experiments = []
    
#     # Define the embedding models and vector search types
#     embedding_models = [
#         "sentence-transformers/all-MiniLM-L6-v2",
#         "BAAI/bge-large-en"
#         # "openai/text-embedding-3-large"
#     ]
    
#     vector_search_types = [
#         "faiss_flat",
#         "faiss_ivf",
#         "faiss_pq",
#         "faiss_hnsw",
#         "faiss_ivf_pq"
#     ]
    
#     # Create all combinations
#     for embedding_model in embedding_models:
#         for vector_search_type in vector_search_types:
#             config = {
#                 "embedding_model": embedding_model,
#                 "vector_search_type": vector_search_type
#             }
#             # Add OpenAI API key if needed
#             if "openai" in embedding_model.lower():
#                 config["openai_api_key"] = "your-openai-api-key-here"
            
#             experiments.append(config)
    
#     # Run experiments
#     for config in experiments:
#         try:
#             exp = RAGExperiment(**config)
#             exp.run_experiment(test_queries)
#         except Exception as e:
#             print(f"Experiment failed for {config}: {str(e)}")
#             continue

# rag_pipeline.py
import os
import time
import json
import logging
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import torch


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
        logging.info(f"RAGExperiment initialized with ID: {self.experiment_id}")
        logging.info(f"Configuration: embedding_model={embedding_model}, vector_search_type={vector_search_type}")
        logging.info(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        

    def load_documents(self):
        """Load text documents and their metadata from the specified directory."""
        start_time = time.time()
        logging.info(f"Loading documents from {self.text_dir}...")
        
        try:
            text_loader = DirectoryLoader(self.text_dir, glob="*.txt", loader_cls=TextLoader)
            documents = text_loader.load()
            logging.info(f"Loaded {len(documents)} documents from text files")
            
            # Try to load metadata if available
            metadata_dir = os.path.join(self.text_dir, "metadata")
            metadata_count = 0
            
            if os.path.exists(metadata_dir):
                logging.info(f"Looking for metadata in {metadata_dir}")
                for doc in documents:
                    txt_filename = os.path.basename(doc.metadata['source'])
                    json_filename = txt_filename.replace('.txt', '_metadata.json')
                    metadata_path = os.path.join(metadata_dir, json_filename)
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            doc.metadata.update(metadata)
                            metadata_count += 1
                
                logging.info(f"Found metadata for {metadata_count} documents")
            else:
                logging.info(f"No metadata directory found at {metadata_dir}")
        
            load_time = time.time() - start_time
            logging.info(f"Document loading completed in {load_time:.2f} seconds")
            return documents, load_time
            
        except Exception as e:
            logging.error(f"Error loading documents: {str(e)}", exc_info=True)
            raise
    
    def chunk_documents(self, documents):
        """Split documents into chunks for embedding and retrieval."""
        start_time = time.time()
        logging.info(f"Chunking {len(documents)} documents with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        
        try:
            text_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            chunks = []
            for doc_idx, doc in enumerate(documents):
                doc_chunks = text_splitter.split_documents([doc])
                logging.debug(f"Document {doc_idx+1}/{len(documents)} split into {len(doc_chunks)} chunks")
                
                for chunk in doc_chunks:
                    chunk.metadata.update({
                        "source_document": doc.metadata.get('filename', ''),
                        "total_pages": doc.metadata.get('total_pages', 0)
                    })
                chunks.extend(doc_chunks)
            
            chunk_time = time.time() - start_time
            avg_chunk_len = sum(len(c.page_content.split()) for c in chunks) / max(1, len(chunks))
            
            logging.info(f"Created {len(chunks)} chunks in {chunk_time:.2f} seconds")
            logging.info(f"Average chunk length: {avg_chunk_len:.1f} tokens")
            return chunks, chunk_time
            
        except Exception as e:
            logging.error(f"Error chunking documents: {str(e)}", exc_info=True)
            raise
    
    def create_embeddings(self):
        """Initialize the embedding model based on configuration."""
        start_time = time.time()
        logging.info(f"Loading embedding model: {self.embedding_model}")
        
        try:
            if "bge-large" in self.embedding_model.lower():
                from langchain_community.embeddings import HuggingFaceBgeEmbeddings
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logging.info(f"Using BGE embeddings on device: {device}")
            
                embeddings = HuggingFaceBgeEmbeddings(
                    model_name="BAAI/bge-large-en",
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True}
                )
            else:
                # default to sentence-transformers
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logging.info(f"Using HuggingFace embeddings on device: {device}")
                
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': device}
                )
            
            # Test the embedding model with a sample
            sample_emb = embeddings.embed_query("test sample")
            emb_dim = len(sample_emb)
            
            embedding_time = time.time() - start_time
            logging.info(f"Embedding model loaded in {embedding_time:.2f} seconds")
            logging.info(f"Embedding dimension: {emb_dim}")
            return embeddings, embedding_time
            
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            raise
    
    def create_faiss_index(self, dimension):
        """Create the appropriate FAISS index based on vector_search_type."""
        logging.info(f"Creating FAISS index of type {self.vector_search_type} with dimension {dimension}")
        
        try:
            if self.vector_search_type == "faiss_flat":
                logging.info("Creating FAISS Flat index (exact search)")
                index = faiss.IndexFlatL2(dimension)
            
            elif self.vector_search_type == "faiss_ivf":
                nlist = 100  # no of clusters
                logging.info(f"Creating FAISS IVF index with nlist={nlist}")
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            elif self.vector_search_type == "faiss_pq":
                m = 8  # no of subquantizers
                nbits = 8  # bits per subquantizer
                logging.info(f"Creating FAISS PQ index with m={m}, nbits={nbits}")
                index = faiss.IndexPQ(dimension, m, nbits)
            
            elif self.vector_search_type == "faiss_hnsw":
                M = 32  # no of connections per element
                logging.info(f"Creating FAISS HNSW index with M={M}")
                index = faiss.IndexHNSWFlat(dimension, M)
            
            elif self.vector_search_type == "faiss_ivf_pq":
                nlist = 100
                m = 8
                nbits = 8
                logging.info(f"Creating FAISS IVF-PQ index with nlist={nlist}, m={m}, nbits={nbits}")
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
            
            else:
                err_msg = f"Unsupported vector search type: {self.vector_search_type}"
                logging.error(err_msg)
                raise ValueError(err_msg)
            
            logging.info(f"FAISS index created successfully")
            return index
            
        except Exception as e:
            logging.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
            raise
    
    def create_vector_store(self, chunks, embeddings):
        """
        Create a LangChain FAISS vector store using a manually built FAISS index,
        with correct parameter tuning on the raw index object.
        """
        import time, logging, faiss
        from langchain_community.vectorstores import FAISS

        start_time = time.time()
        logging.info(f"Creating vector store for {len(chunks)} chunks")

        # 1) Determine embedding dimensionality
        sample_emb = embeddings.embed_query("test")
        dimension = len(sample_emb)
        logging.info(f"Embedding dimension: {dimension}")

        # 2) Build the raw FAISS index
        faiss_index = self.create_faiss_index(dimension)

        # ——— FIX: Tune raw faiss_index before adding vectors ———
        if isinstance(faiss_index, faiss.IndexIVFFlat):
            logging.info("Tuning IVF: setting nprobe=10")
            faiss_index.nprobe = 10
        elif isinstance(faiss_index, faiss.IndexHNSWFlat):
            logging.info("Tuning HNSW: setting efConstruction=200, efSearch=200")
            faiss_index.hnsw.efConstruction = 200
            faiss_index.hnsw.efSearch       = 200
        # ————————————————————————————————————————————————

        # 3) Generate embeddings for all chunks in batches
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.page_content for c in batch]
            if hasattr(embeddings, "embed_documents"):
                batch_emb = embeddings.embed_documents(texts)
            else:
                batch_emb = [embeddings.embed_query(t) for t in texts]
            all_embeddings.extend(batch_emb)
            logging.debug(f"  Embedded {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")

        all_embeddings = np.vstack(all_embeddings).astype("float32")

        # 4) Train (if needed) and add to FAISS index
        if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
            logging.info("Training FAISS index")
            faiss_index.train(all_embeddings)
        logging.info(f"Adding {all_embeddings.shape[0]} vectors to FAISS index")
        faiss_index.add(all_embeddings)

        # 5) Swap into LangChain store
        texts     = [c.page_content for c in chunks]
        metadatas = [c.metadata     for c in chunks]
        vectorstore = FAISS.from_texts(
            texts,
            embeddings,
            metadatas=metadatas
        )
        vectorstore.index = faiss_index

        index_time = time.time() - start_time
        logging.info(f"Vector store created in {index_time:.2f}s")
        return vectorstore, index_time


    def create_dummy_vectorstore(self, embedding_model):
        """Create a simple FAISS vectorstore just for baseline testing."""
        logging.info("Creating dummy vectorstore for baseline testing")
        try:
            dummy_store = FAISS.from_texts(
                ["dummy text"],
                embedding_model
            )
            logging.info("Dummy vectorstore created successfully")
            return dummy_store
        except Exception as e:
            logging.error(f"Error creating dummy vectorstore: {str(e)}", exc_info=True)
            raise

    

    def create_vector_store_from_embeddings(
        self, embedding_model, texts, metadatas, precomputed_embeddings
        ):
        """
        Create a vector store using precomputed embeddings, with correct
        FAISS parameter tuning on the raw index object.
        """
        # 1) Determine embedding dimensionality
        dimension = precomputed_embeddings.shape[1]

        # 2) Build the FAISS index
        faiss_index = self.create_faiss_index(dimension)

        # ——— FIXED: tune params directly on faiss_index ———
        if isinstance(faiss_index, faiss.IndexIVFFlat):
            logging.info("Setting IVF nprobe=10 on raw faiss_index")
            faiss_index.nprobe = 10
        elif isinstance(faiss_index, faiss.IndexHNSWFlat):
            logging.info("Setting HNSW efConstruction=200, efSearch=200 on raw faiss_index")
            faiss_index.hnsw.efConstruction = 200
            faiss_index.hnsw.efSearch       = 200
        # ——————————————————————————————————————————

        # 3) Train (if needed) and add vectors
        if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
            faiss_index.train(precomputed_embeddings)
        faiss_index.add(precomputed_embeddings)
        logging.info(f"Added {precomputed_embeddings.shape[0]} vectors to FAISS index")

        # 4) Build the LangChain FAISS store and swap in our tuned index
        vectorstore = FAISS.from_texts(
            texts,
            embedding_model,
            metadatas=metadatas
        )
        vectorstore.index = faiss_index

        return vectorstore



# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
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
    ]
    
    vector_search_types = [
        "faiss_flat",
        "faiss_ivf",
        "faiss_pq",
        "faiss_hnsw",
        "faiss_ivf_pq"
    ]
    
    # Log available resources
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            logging.info(f"Device {i}: {device_name} with {total_memory:.2f} GB memory")
    else:
        logging.info("No CUDA devices available, running on CPU")
    
    logging.info(f"Configured to test {len(embedding_models)} embedding models and {len(vector_search_types)} vector search types")
    logging.info(f"Total experiment combinations: {len(embedding_models) * len(vector_search_types)}")