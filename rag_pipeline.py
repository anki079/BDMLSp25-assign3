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
        
        os.makedirs(results_dir, exist_ok=True)
        
        self.experiment_id = f"{embedding_model.split('/')[-1]}_{vector_search_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"RAGExperiment initialized with ID: {self.experiment_id}")
        logging.info(f"Configuration: embedding_model={embedding_model}, vector_search_type={vector_search_type}")
        logging.info(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        

    # load txt documents and metadata from specified directory
    def load_documents(self):
        
        start_time = time.time()
        logging.info(f"Loading documents from {self.text_dir}...")
        
        try:
            text_loader = DirectoryLoader(self.text_dir, glob="*.txt", loader_cls=TextLoader)
            documents = text_loader.load()
            logging.info(f"Loaded {len(documents)} documents from text files")
            
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
    
    # split docs into chunks for embedding and retrieval
    def chunk_documents(self, documents):
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
    
    # initialize embedding model
    def create_embeddings(self):
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
            
            # test with sample
            sample_emb = embeddings.embed_query("test sample")
            emb_dim = len(sample_emb)
            
            embedding_time = time.time() - start_time
            logging.info(f"Embedding model loaded in {embedding_time:.2f} seconds")
            logging.info(f"Embedding dimension: {emb_dim}")
            return embeddings, embedding_time
            
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            raise
    
    # create appropriate faiss index based on vector_search_type
    def create_faiss_index(self, dimension):
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
    

    # create a langchain faiss vector store using a manually built faiss index with correct parameter tuning on the raw index object
    def create_vector_store(self, chunks, embeddings):
        import time, logging, faiss
        from langchain_community.vectorstores import FAISS

        start_time = time.time()
        logging.info(f"Creating vector store for {len(chunks)} chunks")

        sample_emb = embeddings.embed_query("test")
        dimension = len(sample_emb)
        logging.info(f"Embedding dimension: {dimension}")

        # build the raw faiss index
        faiss_index = self.create_faiss_index(dimension)

        # tune raw faiss_index before adding vectors
        if isinstance(faiss_index, faiss.IndexIVFFlat):
            logging.info("Tuning IVF: setting nprobe=10")
            faiss_index.nprobe = 10
        elif isinstance(faiss_index, faiss.IndexHNSWFlat):
            logging.info("Tuning HNSW: setting efConstruction=200, efSearch=200")
            faiss_index.hnsw.efConstruction = 200
            faiss_index.hnsw.efSearch       = 200

        # batch generate embeddings for all chunks
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

        if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
            logging.info("Training FAISS index")
            faiss_index.train(all_embeddings)
        logging.info(f"Adding {all_embeddings.shape[0]} vectors to FAISS index")
        faiss_index.add(all_embeddings)

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


    # simple faiss vectorstore for baseline testing
    def create_dummy_vectorstore(self, embedding_model):
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

    
        # create vector store using precomputed embeddings
    def create_vector_store_from_embeddings(
        self, embedding_model, texts, metadatas, precomputed_embeddings
        ):

        dimension = precomputed_embeddings.shape[1]

        faiss_index = self.create_faiss_index(dimension)

        if isinstance(faiss_index, faiss.IndexIVFFlat):
            logging.info("Setting IVF nprobe=10 on raw faiss_index")
            faiss_index.nprobe = 10
        elif isinstance(faiss_index, faiss.IndexHNSWFlat):
            logging.info("Setting HNSW efConstruction=200, efSearch=200 on raw faiss_index")
            faiss_index.hnsw.efConstruction = 200
            faiss_index.hnsw.efSearch       = 200

        if hasattr(faiss_index, "is_trained") and not faiss_index.is_trained:
            faiss_index.train(precomputed_embeddings)
        faiss_index.add(precomputed_embeddings)
        logging.info(f"Added {precomputed_embeddings.shape[0]} vectors to FAISS index")

        vectorstore = FAISS.from_texts(
            texts,
            embedding_model,
            metadatas=metadatas
        )
        vectorstore.index = faiss_index

        return vectorstore



# example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    test_queries = [
        "What are the main causes of climate change?",
        "How does global warming affect ocean levels?",
        "What are renewable energy solutions?"
    ]
    
    experiments = []
    
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
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logging.info(f"Device {i}: {device_name} with {total_memory:.2f} GB memory")
    else:
        logging.info("No CUDA devices available, running on CPU")
    
    logging.info(f"Configured to test {len(embedding_models)} embedding models and {len(vector_search_types)} vector search types")
    logging.info(f"Total experiment combinations: {len(embedding_models) * len(vector_search_types)}")