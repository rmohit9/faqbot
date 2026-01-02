
import os
import sys
from pathlib import Path
import django
import logging

# Setup environment
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'faqbackend.settings')
django.setup()

from faq.rag.components.vector_store.vector_store import VectorStore
from faq.rag.components.vector_store.chroma_store import ChromaVectorStore
from faq.rag.config.settings import rag_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    print("="*50)
    print("MIGRATING VECTORS TO CHROMADB")
    print("="*50)

    # 1. Initialize Old Store
    old_path = rag_config.config.vector_store_path
    print(f"Loading from Pickle store at: {old_path}")
    
    old_store = VectorStore(storage_path=old_path)
    
    if not old_store._vectors:
        print("No vectors found in old store! Exiting.")
        return

    print(f"Found {len(old_store._vectors)} vectors to migrate.")

    # 2. Initialize New Store
    chroma_path = old_path
    if "chroma" not in chroma_path:
        chroma_path = f"{chroma_path}_chroma"
        
    print(f"Initializing ChromaDB store at: {chroma_path}")
    try:
        new_store = ChromaVectorStore(storage_path=chroma_path)
    except ImportError:
        print("Error: ChromaDB library not found. Please run 'pip install chromadb'")
        return
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    # 3. Migrate Data
    print("\nStarting migration...")
    
    vectors_to_migrate = []
    
    # Iterate over old metadata to reconstruct FAQEntry objects
    # Note: old_store._vectors has embeddings, old_store._metadata has FAQEntry (without embedding potentially)
    # We need to attach embeddings to FAQEntry before sending to store_vectors works
    
    count = 0
    for faq_id, entry in old_store._metadata.items():
        if faq_id in old_store._vectors:
            entry.embedding = old_store._vectors[faq_id]
            vectors_to_migrate.append(entry)
            count += 1
            
    print(f"Prepared {count} entries for migration.")
    
    try:
        # We can pass None for document_id since we are migrating individual vectors
        # However, to preserve document tracking, we might want to group by document
        # But for now, we'll just bulk store them as individual items to get them in index.
        # Document tracking will be rebuilt on next ingestion or we can try to migrate it too.
        
        # Let's try to preserve document associations if possible
        doc_faqs = old_store._document_faqs
        
        # Invert map for easier lookup if needed, but since store_vectors takes a list,
        # we can just dump all. The new store will track them if we provide doc info, 
        # but store_vectors(vectors) without doc_id doesn't update doc tracking in new store implementation 
        # unless we modify it to inspect source_document field or we pass doc_id.
        
        # Simple migration: just store vectors. Ingestion pipeline usually handles doc tracking.
        # If we want to fully restore state, we should migrate doc tracking too.
        # But doing it correctly requires looping through docs.
        
        if doc_faqs:
            print("\nMigrating by document groups to preserve tracking...")
            for doc_id, faq_ids in doc_faqs.items():
                doc_hash = old_store._document_hashes.get(doc_id)
                doc_vectors = []
                for fid in faq_ids:
                    if fid in old_store._vectors and fid in old_store._metadata:
                        entry = old_store._metadata[fid]
                        entry.embedding = old_store._vectors[fid]
                        doc_vectors.append(entry)
                
                if doc_vectors:
                    print(f"Migrating document {doc_id} ({len(doc_vectors)} vectors)...")
                    new_store.store_vectors(doc_vectors, document_id=doc_id, document_hash=doc_hash)
                    
                    # Remove from main list so we don't duplicate (though store_vectors handles upsert)
                    # Actually, let's just do documents first, then any leftovers.
                    for v in doc_vectors:
                        if v in vectors_to_migrate:
                             vectors_to_migrate.remove(v) # This might be slow if list is huge
        
        # Migrate remaining (orphaned?) vectors
        remaining = [v for v in vectors_to_migrate if v.id not in new_store.collection.get()['ids']]
        if remaining:
             print(f"\nMigrating {len(remaining)} remaining vectors...")
             new_store.store_vectors(remaining)

        print("\n" + "="*50)
        print("MIGRATION COMPLETE!")
        print("="*50)
        print(f"Total vectors in ChromaDB: {new_store.collection.count()}")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    migrate()
