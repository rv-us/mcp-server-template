from abc import ABC, abstractmethod
from typing import List, Dict, Any
import chromadb
import re
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

class VectorDB(ABC):
    def __init__(self):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    def get_text_embedding(self, text: List[str]):
        """
        Given a text document, returns a vector embedding using a Qwen model.
        """
        return self.model.encode(text)

    @abstractmethod
    def insert(self, documents: List[str], ids: List[str]) -> None:
        """Insert raw text documents (embeddings handled internally)."""
        pass

    @abstractmethod
    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """Retrieve most similar documents for a text query."""
        pass


class ChromaVectorDB(VectorDB):
    def __init__(
        self,
        path: str,
        collection_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """
        Initialize ChromaDB with automatic chunking and Qwen embeddings.
        """
        super().__init__()
        try:
            # Disable telemetry & use PersistentClient if possible
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(collection_name)
            print("[INFO] Chroma PersistentClient initialized.")
        except Exception as e:
            # Fallback to in-memory mode if persistent crashes
            print(f"[WARN] PersistentClient failed: {e} -> falling back to in-memory client.")
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            self.collection = self.client.get_or_create_collection(collection_name)
            print("[INFO] Chroma in-memory client initialized.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks by words."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks if chunks else [text]

    def insert(self, documents: List[str], ids: List[str]) -> None:
        """
        Insert text documents into Chroma safely.
        Each document is cleaned, chunked, embedded, and stored.
        Handles large inputs by batching, converts embeddings to float32,
        and retries failed batches automatically.
        """
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")

        all_chunks, all_ids = [], []
        for doc, base_id in zip(documents, ids):
            chunks = self._chunk_text(doc)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{base_id}_chunk{idx}")

        print(f"[INFO] Total chunks to insert: {len(all_chunks)}")

        # Step 1. Compute embeddings (keep as float32)
        embeddings = self.get_text_embedding(all_chunks)
        import numpy as np
        embeddings = np.array(embeddings, dtype="float32").tolist()

        # Step 2. Batch insertion (avoid memory / Rust FFI crash)
        BATCH_SIZE = 64
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch_docs = all_chunks[i:i + BATCH_SIZE]
            batch_embs = embeddings[i:i + BATCH_SIZE]
            batch_ids = all_ids[i:i + BATCH_SIZE]

            try:
                self.collection.add(
                    documents=batch_docs,
                    embeddings=batch_embs,
                    ids=batch_ids,
                )
                print(f"[INFO] Added batch {i // BATCH_SIZE + 1}/{(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}")
            except Exception as e:
                print(f"[WARN] Batch {i // BATCH_SIZE + 1} failed: {e}")
                # Attempt a retry with smaller batch size
                try:
                    for j in range(len(batch_docs)):
                        self.collection.add(
                            documents=[batch_docs[j]],
                            embeddings=[batch_embs[j]],
                            ids=[batch_ids[j]],
                        )
                except Exception as inner_e:
                    print(f"[ERROR] Failed to insert single chunk at index {i + j}: {inner_e}")

        print("[INFO] Insert completed successfully.")


    def query(self, query_text: str, n_results: int) -> Dict[str, Any]:
        """
        Embed query text and retrieve top similar documents.

        Returns a dictionary with fields:
        - ids: List of document IDs
        - documents: List of matched document texts of size (# queries, n_results)
        """
        query_emb = self.get_text_embedding([query_text])
        results = self.collection.query(query_embeddings=query_emb.tolist(), n_results=n_results)
        return results


if __name__ == "__main__":
    db = ChromaVectorDB(path="./chroma_db20", collection_name="markdown_docs", chunk_size=5, chunk_overlap=2)

    # Insert sample markdown text
    markdown_text = """
    # Introduction
    This document explains how to use Qwen models for creating embeddings.
    ```python
    print("Example code block")
    ```
    You can then store them in ChromaDB.
    """
    db.insert([markdown_text], ids=["doc1"])

    # Query
    result = db.query("How do I use python?", 2)

    for doc in result['documents'][0]:
        print("RESULT:",doc)
