import logging
from typing import List, Dict, Any
import app.embed as embed

logger = logging.getLogger(__name__)

def retrieve_chunks(index, question: str, document_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks using semantic search with Pinecone"""
    
    logger.info(f"Retrieving chunks for question: {question[:50]}...")
    
    # Extract document_id from the first chunk in the index (if available)
    # For a more robust solution, you might want to store this mapping separately
    document_id = None
    
    # Try to extract document_id from document_path
    if document_path:
        import os
        document_name = os.path.basename(document_path)
        # You might need to implement a more sophisticated mapping here
        # For now, we'll search across all documents if document_id is not found
        
        # Since we can't easily get document_id without additional storage,
        # we'll need to modify the approach slightly
        
        # Get all documents and find the matching one
        try:
            # For Pinecone, we'll use a simple approach:
            # Try to get document_id from recent chunks or use a default search
            
            # First attempt: search without document filter
            initial_results = index.search("", "", top_k=1)  # Get any chunk to find document_id
            if initial_results:
                document_id = initial_results[0]['chunk_data'].get('document_id')
                logger.info(f"Found document_id: {document_id}")
        except:
            logger.warning("Could not determine document_id, searching across all documents")
    
    # If we still don't have document_id, we'll search across all documents
    # This is less ideal but will work for single-document scenarios
    if not document_id:
        document_id = ""  # Empty string means no document filter
        logger.warning("Searching across all documents (no document isolation)")
    
    try:
        # Perform semantic search using Pinecone
        results = embed.search_document(index, question, document_id, top_k)
        
        # Format results for the reasoning module
        relevant_chunks = []
        for result in results:
            chunk_data = result['chunk_data']
            
            relevant_chunks.append({
                'text': chunk_data['text'],
                'title': chunk_data.get('title', 'Document Section'),
                'page_number': chunk_data.get('page_number', 1),
                'type': chunk_data.get('type', 'text'),
                'score': result.get('score', 0.0),
                'chunk_id': chunk_data.get('chunk_id', ''),
                'document_id': chunk_data.get('document_id', ''),
                'retrieval_type': 'semantic'
            })
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks using semantic search")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error during chunk retrieval: {e}")
        return []

def get_document_id_from_path(document_path: str) -> str:
    """Helper function to generate document_id from path"""
    import os
    import hashlib
    import time
    
    if not document_path:
        return f"doc_{int(time.time())}"
    
    file_name = os.path.basename(document_path)
    # Create a simple hash-based ID
    content = f"{file_name}_{int(time.time())}"
    return hashlib.md5(content.encode()).hexdigest()[:12]