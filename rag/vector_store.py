"""
Store vectoriel local avec ChromaDB
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from pathlib import Path


class LocalVectorStore:
    """Vector store local avec ChromaDB"""

    def __init__(
        self,
        persist_directory: str = "data/vectordb",
        collection_name: str = "maroclear_knowledge",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # nomic-embed-text produit des vecteurs que l'on normalise manuellement,
        # donc cosine et dot-product sont équivalents ; on conserve cosine.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Base de connaissances Maroclear",
                "hnsw:space": "cosine",
            },
        )

        print(f"✅ ChromaDB initialisé : {self.collection.count()} documents")

    def clear_collection(self):
        """Vide complètement la collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Base de connaissances Maroclear",
                    "hnsw:space": "cosine",  # ← conservé ici aussi
                },
            )
            print(f"Collection {self.collection_name} nettoyée et réinitialisée")
        except Exception as e:
            print(f"Erreur lors du nettoyage : {e}")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ):
        """Ajoute des documents à la collection."""
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"{len(texts)} documents ajoutés à ChromaDB")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Dict = None,
    ) -> Dict:
        """Recherche sémantique."""
        if isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        return results

    def get_collection_stats(self) -> Dict:
        """Statistiques de la collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": str(self.persist_directory),
        }