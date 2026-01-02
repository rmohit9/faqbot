
"""
ChromaDB Vector Store Implementation

This module provides a ChromaDB-backed implementation of the VectorStoreInterface.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from pathlib import Path
import shutil

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from ....interfaces.base import (
    VectorStoreInterface, 
    FAQEntry, 
    SimilarityMatch
)
from ....utils.ngram_utils import get_ngram_overlap

logger = logging.getLogger(__name__)

class ChromaVectorStore(VectorStoreInterface):
    """
    ChromaDB-backed vector store.
    
    Features:
    - Persistent vector storage using ChromaDB (SQLite + HNSW)
    - Metadata filtering
    - Efficient similarity search
    """
    
    def __init__(self, storage_path: str = "vector_store_data_chroma", collection_name: str = "faq_collection"):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            storage_path: Directory path for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is not installed. Please install it with 'pip install chromadb'.")
            
        self.storage_path = str(Path(storage_path).absolute())
        self.collection_name = collection_name
        
        # Initialize Chroma Client
        try:
            self.client = chromadb.PersistentClient(path=self.storage_path)
            
            # Get or create collection
            # We use a simple embedding function that just passes through valid embeddings if we provide them directly,
            # but Chroma requires an embedding function if we don't provide embeddings. 
            # Since we provide pre-computed embeddings, we can use a dummy or default one.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Use cosine similarity
            )
            
            logger.info(f"ChromaVectorStore initialized at {self.storage_path} with collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
            
        # Cache for document tracking (doc_id -> hash) - lightweight enough to keep in memory/file
        # We can store this in a separate JSON file or a separate collection. 
        # For simplicity, we'll maintain a separate tracking file.
        self._tracking_file = Path(self.storage_path) / "document_tracking.json"
        self._document_hashes: Dict[str, str] = {}
        self._document_faqs: Dict[str, List[str]] = {}
        self._load_tracking_data()

    def _load_tracking_data(self):
        """Load document tracking data from disk."""
        if self._tracking_file.exists():
            try:
                with open(self._tracking_file, 'r') as f:
                    data = json.load(f)
                    self._document_hashes = data.get('hashes', {})
                    self._document_faqs = data.get('faqs', {})
            except Exception as e:
                logger.error(f"Failed to load tracking data: {e}")
    
    def _save_tracking_data(self):
        """Save document tracking data to disk."""
        try:
            data = {
                'hashes': self._document_hashes,
                'faqs': self._document_faqs
            }
            with open(self._tracking_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save tracking data: {e}")

    def store_vectors(self, vectors: List[FAQEntry], document_id: Optional[str] = None, document_hash: Optional[str] = None) -> None:
        """Store FAQ vectors in ChromaDB."""
        if not vectors:
            return

        # Prepare data for Chroma
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        faq_ids_for_this_doc = []

        # Remove existing FAQs for this document if updating
        if document_id and document_id in self._document_faqs:
            self.remove_document(document_id) # This cleans up old entries

        for faq in vectors:
            if faq.embedding is None:
                continue
                
            ids.append(faq.id)
            embeddings.append(faq.embedding.tolist()) # Chroma expects list, not numpy array
            
            # Serialize metadata
            # Chroma metadata must be int, float, str, or bool. No lists/dicts.
            # We'll JSON dump complex fields.
            meta = {
                "question": faq.question,
                "answer": faq.answer,
                "category": faq.category,
                "confidence_score": float(faq.confidence_score),
                "source_document": faq.source_document,
                "created_at": faq.created_at.isoformat(),
                "updated_at": faq.updated_at.isoformat(),
                "audience": faq.audience,
                "intent": faq.intent,
                "condition": faq.condition,
                # Store listsas JSON strings
                "keywords": json.dumps(faq.keywords),
            }
            metadatas.append(meta)
            documents.append(faq.question) # Store question as the document text
            
            if document_id:
                faq_ids_for_this_doc.append(faq.id)

        # Batch add to Chroma
        if ids:
            try:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                
                # Update tracking
                if document_id:
                    self._document_hashes[document_id] = document_hash or ""
                    self._document_faqs[document_id] = faq_ids_for_this_doc
                    self._save_tracking_data()
                
                logger.info(f"Stored {len(ids)} vectors in ChromaDB")
            except Exception as e:
                logger.error(f"Error upserting into ChromaDB: {e}")
                raise

    def search_similar(self, query_vector: np.ndarray, threshold: float = 0.7, top_k: int = 10) -> List[SimilarityMatch]:
        """Search using ChromaDB."""
        if not isinstance(query_vector, list):
            query_vector = query_vector.tolist()
            
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["metadatas", "distances", "documents"]
            )
            
            matches = []
            if results['ids'] and results['ids'][0]:
                for i, faq_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    # Convert cosine distance to similarity score: similarity = 1 - distance
                    # (Chroma usually returns cosine distance for 'cosine' space)
                    similarity = 1.0 - distance
                    
                    if similarity < threshold:
                        continue
                        
                    meta = results['metadatas'][0][i]
                    
                    # Reconstruct FAQEntry
                    faq_entry = self._reconstruct_faq_entry(faq_id, meta)
                    
                    match = SimilarityMatch(
                        faq_entry=faq_entry,
                        similarity_score=similarity,
                        match_type='semantic',
                        matched_components=['embedding']
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _reconstruct_faq_entry(self, faq_id: str, meta: Dict) -> FAQEntry:
        """Helper to reconstruct FAQEntry from Chroma metadata."""
        return FAQEntry(
            id=faq_id,
            question=meta.get('question', ''),
            answer=meta.get('answer', ''),
            keywords=json.loads(meta.get('keywords', '[]')),
            category=meta.get('category', 'general'),
            confidence_score=float(meta.get('confidence_score', 0.0)),
            source_document=meta.get('source_document', ''),
            created_at=datetime.fromisoformat(meta.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(meta.get('updated_at', datetime.now().isoformat())),
            audience=meta.get('audience', 'any'),
            intent=meta.get('intent', 'info'),
            condition=meta.get('condition', 'default')
        )

    def search_by_ngrams(self, query_ngrams: List[str], threshold: float = 0.9) -> List[SimilarityMatch]:
        """
        N-gram search. 
        Note: Chroma is optimized for vectors. For efficient N-gram search on large datasets, 
        we would use a traditional search engine like Elasticsearch. 
        Here, we fetch simplified metadata to simulate it, or just return empty if performance is key.
        For compliance with interface, we'll iterate.
        """
        # Fetch all metadata (WARN: Not scalable for millions of rows, but fine for typical FAQ usage)
        # To optimize, we could batch fetch or use filtering if possible?
        # Let's fetch all IDs and their keywords.
        try:
            # Getting all data items
            count = self.collection.count()
            if count == 0:
                return []
                
            # Limit to reasonable number to avoid OOM
            limit = 2000 
            results = self.collection.get(
                limit=limit,
                include=["metadatas"]
            )
            
            matches = []
            query_ngram_set = set(query_ngrams)
            
            for i, faq_id in enumerate(results['ids']):
                meta = results['metadatas'][i]
                keywords = json.loads(meta.get('keywords', '[]'))
                
                if not keywords:
                    continue
                
                faq_ngram_set = set(keywords)
                overlap = get_ngram_overlap(faq_ngram_set, query_ngram_set)
                
                if overlap >= threshold:
                    faq_entry = self._reconstruct_faq_entry(faq_id, meta)
                    matches.append(SimilarityMatch(
                        faq_entry=faq_entry,
                        similarity_score=overlap,
                        match_type='keyword_ngram',
                        matched_components=['keywords']
                    ))
            
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            return matches

        except Exception as e:
            logger.error(f"N-gram search failed: {e}")
            return []

    def batch_search_similar(self, query_vectors: List[np.ndarray], threshold: float, top_k: int) -> List[List[SimilarityMatch]]:
        """Batch search."""
        if not query_vectors:
            return []
            
        query_embeddings = [v.tolist() for v in query_vectors]
        
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["metadatas", "distances"]
            )
            
            all_matches = []
            
            # results['ids'] is list of list of ids
            for i in range(len(query_embeddings)):
                matches = []
                # Check if we have results for this query
                if i < len(results['ids']):
                    query_ids = results['ids'][i]
                    query_distances = results['distances'][i]
                    query_metas = results['metadatas'][i]
                    
                    for j, faq_id in enumerate(query_ids):
                        dist = query_distances[j]
                        similarity = 1.0 - dist
                        
                        if similarity < threshold:
                            continue
                            
                        faq_entry = self._reconstruct_faq_entry(faq_id, query_metas[j])
                        matches.append(SimilarityMatch(
                            faq_entry=faq_entry,
                            similarity_score=similarity,
                            match_type='semantic',
                            matched_components=['embedding']
                        ))
                
                all_matches.append(matches)
                
            return all_matches
            
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return [[] for _ in query_vectors]

    def search_with_filters(self, query_vector: np.ndarray, threshold: float, top_k: int, 
                           category_filter: Optional[str] = None, 
                           audience_filter: Optional[str] = None,
                           intent_filter: Optional[str] = None,
                           condition_filter: Optional[str] = None,
                           confidence_filter: Optional[float] = None,
                           keyword_filter: Optional[List[str]] = None) -> List[SimilarityMatch]:
        
        # Build Chroma 'where' clause
        conditions = []
        if category_filter and category_filter != 'general':
            conditions.append({"category": category_filter})
        if audience_filter and audience_filter != 'any':
            conditions.append({"audience": audience_filter})
        if intent_filter and intent_filter not in ['information', 'any', 'all']:
             conditions.append({"intent": intent_filter})
        # Note: Condition filter with wildcard is tricky in exact match. We'll skip complex wildcard logic for persistent store for now.
        if condition_filter and condition_filter not in ['*', 'default']:
            conditions.append({"condition": condition_filter})
        
        if confidence_filter:
            conditions.append({"confidence_score": {"$gte": confidence_filter}})
            
        where_clause = {}
        if len(conditions) > 1:
            where_clause = {"$and": conditions}
        elif len(conditions) == 1:
            where_clause = conditions[0]
            
        # Execute query
        if not isinstance(query_vector, list):
            query_vector = query_vector.tolist()
            
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["metadatas", "distances"]
            )
            
            # Process results similar to search_similar
            matches = []
            if results['ids'] and results['ids'][0]:
                for i, faq_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance
                    
                    if similarity < threshold:
                        continue
                    
                    meta = results['metadatas'][0][i]
                    
                    # Manual keyword filter check (since Chroma metadata filter is exact match, and keywords is a JSON string)
                    if keyword_filter:
                        faq_keywords = json.loads(meta.get('keywords', '[]'))
                        faq_keywords = [k.lower() for k in faq_keywords]
                        filter_kws = [k.lower() for k in keyword_filter]
                        if not any(k in faq_keywords for k in filter_kws):
                            continue
                            
                    matches.append(SimilarityMatch(
                        faq_entry=self._reconstruct_faq_entry(faq_id, meta),
                        similarity_score=similarity,
                        match_type='semantic',
                        matched_components=['embedding']
                    ))
            
            return matches

        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            return []

    def search_with_ranking(self, query_vector: np.ndarray, threshold: float, top_k: int,
                           boost_recent: bool, boost_high_confidence: bool) -> List[SimilarityMatch]:
        # Perform standard search then rerank
        matches = self.search_similar(query_vector, threshold, top_k * 2)
        
        for match in matches:
            boost_factor = 1.0
            if boost_recent:
                days_since_update = (datetime.now() - match.faq_entry.updated_at).days
                if days_since_update < 30:
                    boost_factor *= 1.1
            
            if boost_high_confidence and match.faq_entry.confidence_score > 0.8:
                boost_factor *= 1.05
                
            match.similarity_score *= boost_factor
            
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:top_k]

    def update_vector(self, faq_id: str, new_vector: np.ndarray) -> None:
        """Update single vector."""
        # Need to fetch existing metadata first to preserve it?
        # Or assumes we just want to update the vector?
        # Chroma upsert overwrites. We need to be careful.
        # Ideally, we should fetch, update embedding, then upsert.
        
        current = self.collection.get(ids=[faq_id], include=['metadatas', 'documents'])
        if not current['ids']:
            return # Doesn't exist
            
        meta = current['metadatas'][0]
        doc = current['documents'][0]
        meta['updated_at'] = datetime.now().isoformat()
        
        self.collection.upsert(
            ids=[faq_id],
            embeddings=[new_vector.tolist()],
            metadatas=[meta],
            documents=[doc]
        )

    def get_vector_stats(self) -> Dict[str, Any]:
        """Get stats."""
        try:
            count = self.collection.count()
            return {
                "total_vectors": count,
                "storage_type": "chromadb",
                "collection": self.collection_name,
                "path": self.storage_path
            }
        except:
             return {"total_vectors": 0}

    def delete_vector(self, faq_id: str) -> bool:
        try:
            self.collection.delete(ids=[faq_id])
            for doc_id, faqs in self._document_faqs.items():
                if faq_id in faqs:
                    faqs.remove(faq_id)
            self._save_tracking_data()
            return True
        except:
            return False

    def clear_all(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name, metadata={"hnsw:space": "cosine"})
            self._document_hashes = {}
            self._document_faqs = {}
            self._save_tracking_data()
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False

    def is_document_processed(self, document_id: str, document_hash: str) -> bool:
        return self._document_hashes.get(document_id) == document_hash

    def get_document_faqs(self, document_id: str) -> List[str]:
        return self._document_faqs.get(document_id, [])

    def remove_document(self, document_id: str) -> int:
        faqs = self._document_faqs.get(document_id, [])
        if not faqs:
            return 0
        
        try:
            self.collection.delete(ids=faqs)
            del self._document_faqs[document_id]
            if document_id in self._document_hashes:
                del self._document_hashes[document_id]
            self._save_tracking_data()
            return len(faqs)
        except Exception as e:
            logger.error(f"Remove document failed: {e}")
            return 0

    def backup_store(self, backup_path: Optional[str] = None) -> str:
        # Chroma PersistentClient stores data in a directory. Backup is just copying the dir.
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(self.storage_path).parent / f"backup_chroma_{timestamp}"
        else:
            backup_path = Path(backup_path)
            
        try:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(self.storage_path, backup_path)
            return str(backup_path)
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return ""

    def restore_from_backup(self, backup_path: str) -> bool:
        # Copy backup dir to storage path
        try:
            if Path(self.storage_path).exists():
                shutil.rmtree(self.storage_path)
            shutil.copytree(backup_path, self.storage_path)
            
            # Re-init client
            self.client = chromadb.PersistentClient(path=self.storage_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._load_tracking_data()
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
