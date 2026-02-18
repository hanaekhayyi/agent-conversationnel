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
        similarity_threshold: float = 0.5
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        print(f"[INFO] Retriever initialisé avec similarity_threshold={similarity_threshold}")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Récupère les documents pertinents pour une query
        
        Returns:
            Liste de documents avec scores
        """
        
        print(f"[DEBUG] Requête: {query[:100]}...")
        
        # Générer l'embedding de la query
        query_embedding = self.embeddings.embed_query(query)
        print(f"[DEBUG] Embedding query généré: {type(query_embedding)}, shape: {len(query_embedding)}")
        
        # Recherche dans le vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=self.top_k
        )
        
        print(f"[DEBUG] Résultats bruts: {len(results['documents'][0]) if results['documents'] else 0} documents trouvés")
        
        # Formater les résultats
        retrieved_docs = []
        
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                
                # ✅ FORMULE POUR EMBEDDINGS NORMALISÉS :
                # Avec vecteurs normalisés, L2 distance = sqrt(2 - 2*cosine_sim)
                # Donc: cosine_sim = 1 - (distance^2 / 2)
                # Approximation simple : similarity = 1 / (1 + distance)
                similarity_score = 1 / (1 + distance)
                
                print(f"[DEBUG] Doc {i}: distance={distance:.4f}, similarity={similarity_score:.4f}")
                
                # Filtrer par seuil de similarité
                if similarity_score >= self.similarity_threshold:
                    retrieved_docs.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "score": similarity_score,
                        "id": results['ids'][0][i]
                    })
        
        print(f"[DEBUG] Documents retenus avec similarity >= {self.similarity_threshold}: {len(retrieved_docs)}")
        
        return retrieved_docs