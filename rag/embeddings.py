"""
Génération d'embeddings locaux avec nomic-embed-text via Ollama
"""

import ollama
from typing import List
import numpy as np


class LocalEmbeddings:
    """Génère des embeddings en local via Ollama (bge-m3)"""

    def __init__(self, model_name: str = "bge-m3"):
        self.model_name = model_name

        print(f"🔧 Modèle d'embeddings : {model_name} (via Ollama)")

        # Vérifier qu'Ollama est accessible et que le modèle est disponible
        try:
            available = [m["name"] for m in ollama.list()["models"]]
            if not any(self.model_name in m for m in available):
                print(
                    f"⚠️  Modèle '{model_name}' introuvable. "
                    f"Lance : ollama pull {model_name}"
                )
            else:
                print(f"✅ Modèle '{model_name}' disponible")
        except Exception as e:
            print(f"⚠️  Impossible de contacter Ollama : {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, vector: List[float]) -> List[float]:
        """Normalise un vecteur (norme L2 = 1)."""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return vector
        return (arr / norm).tolist()

    def _embed_single(self, text: str) -> List[float]:
        """Appel brut à l'API Ollama pour un texte."""
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]

    # ------------------------------------------------------------------
    # API publique (identique à l'ancienne classe)
    # ------------------------------------------------------------------

    def embed_documents(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """
        Génère des embeddings normalisés pour une liste de documents.

        Le préfixe 'search_document:' améliore la qualité de récupération
        avec nomic-embed-text.
        """
        print(f"🔄 Génération d'embeddings pour {len(texts)} documents...")

        embeddings: List[List[float]] = []

        for i, text in enumerate(texts):
            prefixed = f"search_document: {text}"
            raw = self._embed_single(prefixed)
            embeddings.append(self._normalize(raw))

            if (i + 1) % 10 == 0 or (i + 1) == len(texts):
                print(f"   {i + 1}/{len(texts)} traités", end="\r")

        print()  # saut de ligne après le \r

        # Vérification de la normalisation
        first_norm = np.linalg.norm(embeddings[0])
        print(
            f"✅ Embeddings normalisés "
            f"(norme du 1er vecteur : {first_norm:.4f} ≈ 1.0)"
        )

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Génère un embedding normalisé pour une requête utilisateur.

        Le préfixe 'search_query:' est distinct de 'search_document:'
        afin d'optimiser la similarité asymétrique.
        """
        prefixed = f"search_query: {query}"
        raw = self._embed_single(prefixed)
        return self._normalize(raw)