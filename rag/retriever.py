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
        top_k: int = 5,
        distance_threshold: float = 2.5  # Seuil de distance au lieu de similarité
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.distance_threshold = distance_threshold
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Récupère les documents pertinents pour une query
        
        Returns:
            Liste de documents avec scores
        """
        
        # Générer l'embedding de la query
        query_embedding = self.embeddings.embed_query(query)
        
        print(f"[DEBUG] Embedding query généré: {type(query_embedding)}, shape: {len(query_embedding) if isinstance(query_embedding, list) else 'N/A'}")
        print(f"[DEBUG] Vector store contient {self.vector_store.collection.count()} documents")
        
        # Recherche dans le vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k
        )
        
        print(f"[DEBUG] Résultats bruts: {len(results['documents'][0]) if results['documents'] else 0} documents trouvés")
        
        # Formater les résultats
        retrieved_docs = []
        
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                
                # Convertir distance en similarité normalisée
                # Approche: 1 / (1 + distance) pour que les petites distances = haute similarité
                similarity_score = 1 / (1 + abs(distance))
                
                print(f"[DEBUG] Doc {i}: distance={distance:.4f}, similarity={similarity_score:.4f}")
                
                # Filtrer par seuil sur la distance brute (distances petites = bons matches)
                # Seuil de distance: documents avec distance < threshold
                if distance < self.distance_threshold:
                    retrieved_docs.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "score": similarity_score,
                        "distance": distance,
                        "id": results['ids'][0][i]
                    })
        
        print(f"[DEBUG] Documents retenus avec distance < {self.distance_threshold}: {len(retrieved_docs)}")
        
        return retrieved_docs