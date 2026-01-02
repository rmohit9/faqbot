
import os
import sys
from pathlib import Path
import django
import numpy as np

# Setup environment
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.core import initialize_rag_system
from faq.rag.config.settings import rag_config

def verify():
    print("="*50)
    print("VERIFYING CHROMADB INTEGRATION")
    print("="*50)

    # 1. Check Config
    print(f"Current Vector Store Type: {rag_config.config.vector_store_type}")
    if rag_config.config.vector_store_type != "chroma":
        print("Error: RAG_VECTOR_STORE_TYPE is not set to 'chroma' in .env or config.")
        return

    # 2. Initialize RAG System
    print("\nInitializing RAG system...")
    try:
        rag_system = initialize_rag_system(perform_health_check=True)
        print("✓ RAG system initialized successfully with ChromaDB.")
    except Exception as e:
        print(f"✗ Failed to initialize RAG system: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Check Vector Store
    store = rag_system.vector_store
    print(f"\nVector Store Details:")
    print(f"  Type: {type(store).__name__}")
    
    stats = store.get_vector_stats()
    print(f"  Total Vectors: {stats.get('total_vectors', 0)}")
    
    if stats.get('total_vectors', 0) == 0:
        print("Warning: No vectors found in ChromaDB. Migration might not have worked or path is wrong.")
    else:
        print("✓ Successfully found migrated vectors in ChromaDB.")

    # 4. Test Search (Optional)
    print("\nPerforming test search...")
    test_query = "How do I reset my password?"
    results = rag_system.answer_query(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {results.text[:100]}...")
    print(f"Confidence: {results.confidence}")
    print(f"Source FAQs used: {len(results.source_faqs)}")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    verify()
