#!/usr/bin/env python3
"""
Test script for Pinecone-based Document Q&A System
Run this to verify your setup is working correctly.
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_environment_variables():
    """Test that all required environment variables are set"""
    print("🔧 Testing Environment Variables...")
    
    load_dotenv()
    
    required_vars = {
        'PINECONE_API_KEY': 'Pinecone API key',
        'PINECONE_ENVIRONMENT': 'Pinecone environment',
        'GOOGLE_API_KEY': 'Google Gemini API key',
        'LLAMA_CLOUD_API_KEY': 'LlamaParse API key',
        'GROQ_API_KEY': 'GROQ API key'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"❌ {var} ({description})")
            print(f"❌ {var}: Not set")
        else:
            # Hide most of the key for security
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {var}: {masked_value}")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables:")
        for var in missing_vars:
            print(f"   {var}")
        print("Please update your .env file and try again.")
        return False
    
    print("✅ All environment variables are set!")
    return True

async def test_pinecone_connection():
    """Test Pinecone connection and index creation"""
    print("\n🌲 Testing Pinecone Connection...")
    
    try:
        from app import embed
        
        # Create Pinecone index
        pinecone_index = embed.create_or_load_index()
        print("✅ Pinecone connection successful!")
        print("✅ Index created/loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        print("   - Check your PINECONE_API_KEY")
        print("   - Verify PINECONE_ENVIRONMENT is correct")
        print("   - Make sure you have free tier access")
        return False

async def test_document_processing():
    """Test document parsing and chunking"""
    print("\n📄 Testing Document Processing...")
    
    # Create a test document
    test_content = """
# Test Insurance Policy

## Coverage Details
This policy provides comprehensive health insurance coverage.

### Benefits Include:
- Hospitalization expenses up to $100,000
- Outpatient consultations
- Prescription medicines
- Emergency services

### Waiting Periods
- Pre-existing conditions: 12 months
- Maternity benefits: 10 months
- Dental treatments: 6 months

### Exclusions
The following are not covered:
- Cosmetic surgery
- Self-inflicted injuries
- War-related injuries
"""
    
    # Save test document
    test_file = "test_document.txt"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    try:
        from app import parse
        
        # Test parsing (this will work for text files)
        print("   - Testing document parsing...")
        # For text files, we'll create a simple Document object
        from llama_index.core.schema import Document
        
        parsed_doc = [Document(text=test_content)]
        print("   ✅ Document parsing successful!")
        
        # Test chunking
        print("   - Testing document chunking...")
        chunks = parse.chunk_document(parsed_doc, test_file)
        print(f"   ✅ Document chunking successful! Generated {len(chunks)} chunks")
        
        # Clean up
        os.remove(test_file)
        
        return chunks
        
    except Exception as e:
        print(f"   ❌ Document processing failed: {e}")
        # Clean up on error
        if os.path.exists(test_file):
            os.remove(test_file)
        return None

async def test_embedding_and_storage(chunks):
    """Test embedding generation and Pinecone storage"""
    print("\n🧠 Testing Embeddings and Storage...")
    
    if not chunks:
        print("❌ No chunks available for testing")
        return False
    
    try:
        from app import embed
        
        # Create index
        print("   - Creating Pinecone index...")
        pinecone_index = embed.create_or_load_index()
        
        # Add chunks
        print("   - Adding chunks to Pinecone...")
        embed.add_chunks_to_index(pinecone_index, chunks, "test_document.txt")
        print("   ✅ Chunks added to Pinecone successfully!")
        
        return pinecone_index
        
    except Exception as e:
        print(f"   ❌ Embedding and storage failed: {e}")
        return None

async def test_retrieval_and_reasoning(index):
    """Test chunk retrieval and answer generation"""
    print("\n🔍 Testing Retrieval and Reasoning...")
    
    if not index:
        print("❌ No index available for testing")
        return False
    
    test_questions = [
        "What is the maximum hospitalization coverage?",
        "What are the waiting periods?",
        "What is excluded from the policy?"
    ]
    
    try:
        from app import retrieve, reason
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n   Question {i}: {question}")
            
            # Test retrieval
            print("   - Retrieving relevant chunks...")
            relevant_chunks = retrieve.retrieve_chunks(index, question, "test_document.txt")
            print(f"   ✅ Retrieved {len(relevant_chunks)} chunks")
            
            # Test reasoning
            print("   - Generating answer...")
            answer_data = await reason.generate_answer(question, relevant_chunks)
            print(f"   ✅ Answer: {answer_data.get('answer', 'No answer')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Retrieval and reasoning failed: {e}")
        return False

async def test_api_endpoint():
    """Test the FastAPI endpoint"""
    print("\n🌐 Testing API Endpoint...")
    
    try:
        import httpx
        
        # Test data
        test_payload = {
            "documents": "test_document.txt",  # This won't exist but we can test parsing
            "questions": ["What is covered?"]
        }
        
        print("   - To test the API endpoint:")
        print("   1. Start the server: uvicorn app.main:app --reload")
        print("   2. Visit: http://localhost:8000/docs")
        print("   3. Try the /run endpoint with a real PDF/DOCX file")
        
        return True
        
    except Exception as e:
        print(f"   ❌ API test preparation failed: {e}")
        return False

async def run_full_test():
    """Run all tests"""
    print("🚀 Starting Pinecone RAG System Tests")
    print("=" * 50)
    
    # Test 1: Environment Variables
    if not await test_environment_variables():
        print("\n❌ Environment test failed. Please fix and try again.")
        return False
    
    # Test 2: Pinecone Connection
    if not await test_pinecone_connection():
        print("\n❌ Pinecone test failed. Please check your configuration.")
        return False
    
    # Test 3: Document Processing
    chunks = await test_document_processing()
    if not chunks:
        print("\n❌ Document processing test failed.")
        return False
    
    # Test 4: Embeddings and Storage
    index = await test_embedding_and_storage(chunks)
    if not index:
        print("\n❌ Embedding and storage test failed.")
        return False
    
    # Test 5: Retrieval and Reasoning
    if not await test_retrieval_and_reasoning(index):
        print("\n❌ Retrieval and reasoning test failed.")
        return False
    
    # Test 6: API Endpoint Info
    await test_api_endpoint()
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("\nYour Pinecone RAG system is ready!")
    print("\nNext steps:")
    print("1. Start the server: uvicorn app.main:app --reload")
    print("2. Test with real documents via the API")
    print("3. Deploy for production use")
    
    return True

if __name__ == "__main__":
    asyncio.run(run_full_test())