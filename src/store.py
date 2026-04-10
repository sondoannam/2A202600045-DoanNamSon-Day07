from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self.client = chromadb.Client()
            try:
                self.client.delete_collection(name=self._collection_name)
            except Exception:
                pass
            self._collection = self.client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        
        scored_records = []
        for record in records:
            score = compute_similarity(query_embedding, record["embedding"])
            scored_records.append((score, record))
        
        scored_records.sort(key=lambda x: x[0], reverse=True)
        
        return [record for score, record in scored_records[:top_k]]

    def _format_chroma_results(self, results: dict) -> list[dict[str, Any]]:
        # Standalize output format of ChromaDB to match test suite
        formatted = []
        if not results.get("ids") or not results["ids"][0]:
            return formatted
            
        for i in range(len(results["ids"][0])):
            # Chroma return distance (smaller is better). We reverse it to score (larger is better)
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] or {},
                "score": 1.0 - distance 
            })
            
        formatted.sort(key=lambda x: x["score"], reverse=True)
        return formatted

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return
            
        if self._use_chroma:
            import uuid
            # Generate unique IDs for ChromaDB to permit adding the same logical document ID multiple times
            ids = [f"{doc.id}_{uuid.uuid4().hex}" for doc in docs]
            documents = [doc.content for doc in docs]
            
            # Inject doc_id into metadata for deletion, and ensure we don't pass an empty dict directly if None
            metadatas = []
            for doc in docs:
                m = dict(doc.metadata) if doc.metadata else {}
                m["doc_id"] = doc.id
                metadatas.append(m)
                
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))
                self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if not self._use_chroma:
            # In-memory fallback
            return self._search_records(query, self._store, top_k)
            
        # ChromaDB search
        results = self._collection.query(
            query_embeddings=[self._embedding_fn(query)],
            n_results=top_k
        )
        
        return self._format_chroma_results(results)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k)
            
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=metadata_filter
            )
            return self._format_chroma_results(results)
        else:
            filtered_records = []
            for record in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if record["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            return self._search_records(query, filtered_records, top_k)

    def _matches_filter(self, record: dict, metadata_filter: dict) -> bool:
        if not metadata_filter:
            return True
        
        record_meta = record.get("metadata", {})
        for key, value in metadata_filter.items():
            if record_meta.get(key) != value:
                return False
        return True

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            initial_size = self._collection.count()
            try:
                # Delete by document ID
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass
            try:
                # Delete by doc_id in metadata (for chunks)
                self._collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass
            return self._collection.count() < initial_size
        else:
            initial_size = len(self._store)
            self._store = [
                r for r in self._store 
                if r.get("id") != doc_id and r.get("metadata", {}).get("doc_id") != doc_id
            ]
            return len(self._store) < initial_size
        
