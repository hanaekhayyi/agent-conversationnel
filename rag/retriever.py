"""
Retriever qui combine vector store et reranking
"""

from typing import List, Dict
from rag.vector_store import LocalVectorStore
from rag.embeddings import LocalEmbeddings


class Retriever:
    """Retrieval sémantique avec score filtering"""

    def __init__(
        self,
        vector_store: LocalVectorStore,
        embeddings: LocalEmbeddings,
        top_k: int = 8,                  # ← augmenté de 5 à 8
        similarity_threshold: float = 0.4,  # ← abaissé de 0.5 à 0.4
    ):
        self.vector_store         = vector_store
        self.embeddings           = embeddings
        self.top_k                = top_k
        self.similarity_threshold = similarity_threshold

        print(
            f"[INFO] Retriever initialisé — "
            f"top_k={top_k}, similarity_threshold={similarity_threshold}"
        )

    def retrieve(self, query: str) -> List[Dict]:
        """
        Récupère les documents pertinents pour une query.

        Stratégie :
        1. On récupère top_k * 2 candidats bruts pour ne pas rater
           un bon chunk (ChromaDB peut classer imparfaitement).
        2. On filtre sur similarity_threshold.
        3. On retourne au maximum top_k documents.
        """
        print(f"[DEBUG] Requête : {query[:100]}...")

        query_embedding = self.embeddings.embed_query(query)
        print(f"[DEBUG] Embedding généré — dim={len(query_embedding)}")

        # Récupérer plus de candidats que nécessaire pour le filtrage
        fetch_k = self.top_k * 2
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=fetch_k,
        )

        raw_count = len(results["documents"][0]) if results.get("documents") else 0
        print(f"[DEBUG] Candidats bruts : {raw_count}")

        retrieved_docs: List[Dict] = []

        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                cosine_distance = results["distances"][0][i]
                similarity      = 1.0 - cosine_distance

                meta = results["metadatas"][0][i]
                doc_type = meta.get("type", "?")
                source   = meta.get("source", "?")

                print(
                    f"[DEBUG] #{i:02d} sim={similarity:.4f} "
                    f"type={doc_type:<12} source={source[:40]}"
                )

                if similarity >= self.similarity_threshold:
                    retrieved_docs.append({
                        "content":  doc,
                        "metadata": meta,
                        "score":    similarity,
                        "id":       results["ids"][0][i],
                    })

        # Limiter au top_k après filtrage
        retrieved_docs = retrieved_docs[:self.top_k]

        print(
            f"[DEBUG] Retenus (sim ≥ {self.similarity_threshold}) : "
            f"{len(retrieved_docs)}/{raw_count}"
        )

        # Avertissement si trop peu de résultats
        if len(retrieved_docs) == 0:
            print(
                "[WARN] Aucun document retenu ! "
                "Vérifiez le threshold ou relancez l'indexation."
            )
        elif len(retrieved_docs) < 3:
            print(
                f"[WARN] Seulement {len(retrieved_docs)} documents retenus. "
                "Envisagez d'abaisser similarity_threshold."
            )

        return retrieved_docs