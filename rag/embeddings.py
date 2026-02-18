"""
Génération d'embeddings locaux avec sentence-transformers
"""

from sentence_transformers import SentenceTransformer
from typing import List
import torch
import numpy as np

class LocalEmbeddings:
    """Génère des embeddings en local"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Chargement du modèle d'embeddings: {model_name}")
        print(f"🖥️  Device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        print(f"✅ Modèle chargé")
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Génère des embeddings NORMALISÉS pour une liste de textes"""
        
        print(f"🔄 Génération d'embeddings pour {len(texts)} documents...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # ✅ CRITIQUE : Active la normalisation
        )
        
        # Vérifier la normalisation
        first_norm = np.linalg.norm(embeddings[0])
        print(f"✅ Embeddings normalisés (norme du 1er vecteur: {first_norm:.4f} ≈ 1.0)")
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Génère un embedding NORMALISÉ pour une requête"""
        
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True  # ✅ CRITIQUE : Active la normalisation
        )
        
        return embedding.tolist()