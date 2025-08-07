import os
import time
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import TextNode
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini embedding model
embed_model = GeminiEmbedding(
    model_name="models/embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Configure global settings
Settings.embed_model = embed_model

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-aws")
PINECONE_INDEX_NAME = "hackrx-documents"

class PineconeRAGIndex:
    """Simple RAG Index using Pinecone for vector storage"""
    
    def __init__(self):
        self.index = None
        self.vector_store = None
        self.storage_context = None
        
    def create_or_load_index(self):
        """Create or connect to Pinecone index"""
        start_time = time.time()
        logger.info("Initializing Pinecone connection...")
        
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            # Create index if it doesn't exist
            if PINECONE_INDEX_NAME not in pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=768,  # Gemini embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            # Connect to Pinecone index
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            
            # Create vector store
            self.vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index,
                namespace="default"
            )
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create or load LlamaIndex
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=self.storage_context
            )
            
            total_time = time.time() - start_time
            logger.info(f"Pinecone RAG index initialized in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise
        
        return self
    
    def add_chunks(self, chunks: List[Dict[str, Any]], document_path: str = None):
        """Add chunks to Pinecone index"""
        start_time = time.time()
        logger.info(f"Adding {len(chunks)} chunks to Pinecone index")
        
        if not chunks:
            logger.warning("No chunks to add to index")
            return self
        
        document_id = chunks[0]['document_id']
        
        # Remove existing chunks for this document
        self.remove_document(document_id)
        
        # Convert chunks to TextNodes
        nodes = []
        for chunk in chunks:
            node = TextNode(
                text=chunk['text'],
                metadata={
                    'document_id': chunk['document_id'],
                    'document_name': chunk.get('document_name', ''),
                    'file_path': chunk.get('file_path', ''),
                    'title': chunk.get('title', ''),
                    'page_number': chunk.get('page_number', 1),
                    'chunk_id': chunk['chunk_id'],
                    'type': chunk.get('type', 'text'),
                    'created_at': chunk.get('created_at', int(time.time()))
                }
            )
            nodes.append(node)
        
        # Add nodes to index
        try:
            self.index.insert_nodes(nodes)
            add_time = time.time() - start_time
            logger.info(f"Successfully added {len(nodes)} chunks to Pinecone in {add_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error adding chunks to Pinecone: {e}")
            raise
        
        return self
    
    def search(self, query: str, document_id: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks in Pinecone"""
        start_time = time.time()
        logger.info(f"Searching Pinecone for query: '{query[:50]}...' in document {document_id}")
        
        try:
            # Create query filter for document isolation
            filters = MetadataFilters(
                filters=[MetadataFilter(key="document_id", value=document_id)]
            )
            
            # Perform similarity search
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=filters
            )
            
            retrieved_nodes = retriever.retrieve(query)
            
            # Format results
            results = []
            for node in retrieved_nodes:
                results.append({
                    'score': node.score,
                    'chunk_data': {
                        'text': node.node.text,
                        'document_id': node.node.metadata.get('document_id'),
                        'document_name': node.node.metadata.get('document_name'),
                        'title': node.node.metadata.get('title'),
                        'page_number': node.node.metadata.get('page_number'),
                        'chunk_id': node.node.metadata.get('chunk_id'),
                        'type': node.node.metadata.get('type'),
                    }
                })
            
            search_time = time.time() - start_time
            logger.info(f"Found {len(results)} relevant chunks in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    def remove_document(self, document_id: str):
        """Remove all chunks for a specific document from Pinecone"""
        try:
            logger.info(f"Removing document {document_id} from Pinecone")
            # Delete by ref_doc_id (required by PineconeVectorStore)
            self.vector_store.delete(ref_doc_id=document_id)
            logger.info(f"Successfully removed document {document_id} from Pinecone")
            
        except Exception as e:
            logger.error(f"Error removing document from Pinecone: {e}")

# Global instance
_pinecone_index = None

def create_or_load_index():
    """Create or load Pinecone RAG index (singleton pattern)"""
    global _pinecone_index
    
    if _pinecone_index is None:
        _pinecone_index = PineconeRAGIndex()
        _pinecone_index.create_or_load_index()
    
    return _pinecone_index

def add_chunks_to_index(index, chunks: List[Dict[str, Any]], document_path: str = None):
    """Add chunks to Pinecone index"""
    if isinstance(index, PineconeRAGIndex):
        return index.add_chunks(chunks, document_path)
    else:
        raise ValueError("Invalid index type. Expected PineconeRAGIndex")

def search_document(index, query: str, document_id: str, top_k: int = 5):
    """Search for relevant chunks in document"""
    if isinstance(index, PineconeRAGIndex):
        return index.search(query, document_id, top_k)
    else:
        raise ValueError("Invalid index type. Expected PineconeRAGIndex")

# Legacy compatibility functions
def add_chunks_to_hybrid_index(index, chunks: List[Dict[str, Any]], document_path: str = None):
    """Legacy function - redirects to normal RAG"""
    return add_chunks_to_index(index, chunks, document_path)

def hybrid_search(index, query: str, document_id: str, top_k: int = 5):
    """Legacy function - redirects to normal search"""
    return search_document(index, query, document_id, top_k)

# For backward compatibility with existing code
HybridRAGIndex = PineconeRAGIndex
