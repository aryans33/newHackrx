#!/usr/bin/env python3
"""
Minimal test to check each import individually
"""

import sys

def test_individual_imports():
    """Test each import individually"""
    
    imports_to_test = [
        ("os", "os"),
        ("dotenv", "from dotenv import load_dotenv"),
        ("google.generativeai", "import google.generativeai as genai"),
        ("pinecone", "from pinecone import Pinecone"),
        ("llama_index.core", "from llama_index.core import VectorStoreIndex"),
        ("fastapi", "from fastapi import FastAPI"),
    ]
    
    print("🔍 Testing Individual Imports:")
    print("-" * 40)
    
    failed_imports = []
    
    for name, import_statement in imports_to_test:
        try:
            exec(import_statement)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
        except Exception as e:
            print(f"⚠️  {name}: {e}")
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        print("\nInstall commands:")
        
        install_map = {
            "google.generativeai": "pip install google-generativeai",
            "pinecone": "pip install pinecone-client",
            "llama_index.core": "pip install llama-index",
            "fastapi": "pip install fastapi",
            "dotenv": "pip install python-dotenv"
        }
        
        for failed in failed_imports:
            if failed in install_map:
                print(f"  {install_map[failed]}")
    else:
        print("\n🎉 All basic imports successful!")
        return True
    
    return len(failed_imports) == 0

def test_env_file():
    """Test if .env file exists and has required keys"""
    print("\n📁 Testing .env File:")
    print("-" * 20)
    
    import os
    
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("Create a .env file with your API keys")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_keys = ['PINECONE_API_KEY', 'GOOGLE_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            value = os.getenv(key)
            if value:
                print(f"✅ {key}: Set")
            else:
                print(f"❌ {key}: Missing")
                missing_keys.append(key)
        
        return len(missing_keys) == 0
        
    except Exception as e:
        print(f"❌ Error loading .env: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Minimal System Test")
    print("=" * 30)
    
    # Test imports
    imports_ok = test_individual_imports()
    
    # Test .env
    env_ok = test_env_file()
    
    if imports_ok and env_ok:
        print("\n🎉 Basic setup looks good!")
        print("Try running: python quick_test.py")
    else:
        print("\n❌ Issues found. Fix the above and try again.")