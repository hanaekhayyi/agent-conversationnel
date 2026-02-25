"""
Client pour Ollama en local
"""

import ollama
from typing import Dict, Any


class OllamaClient:
    """Client Ollama local"""

    def __init__(self, model: str = "mistral:7b-instruct", temperature: float = 0.1):
        # température abaissée à 0.1 pour réduire les hallucinations
        self.model       = model
        self.temperature = temperature

        try:
            ollama.list()
            print(f"Ollama connecté - Modèle: {model}")
        except Exception as e:
            print(f"Erreur Ollama: {e}")
            raise

    def generate_rag_response(self, query: str, context: str) -> str:
        """Génère une réponse basée sur le contexte fourni"""

        # ── Prompt système ────────────────────────────────────────────────────
        # Principes :
        # - Contrainte stricte : répondre UNIQUEMENT à partir du contexte
        # - Interdire explicitement l'invention ("ne jamais inventer")
        # - Demander de citer la source (glossaire vs document)
        # - Garder un format clair et direct
        system_prompt = """Tu es un assistant spécialisé sur Maroclear, le dépositaire central des titres au Maroc.

RÈGLES ABSOLUES — tu dois les respecter sans exception :
1. Réponds UNIQUEMENT en utilisant les informations contenues dans le CONTEXTE fourni.
2. Si l'information demandée n'est PAS dans le contexte, réponds exactement :
   "Je n'ai pas cette information dans les documents disponibles."
   Ne tente JAMAIS de compléter avec tes connaissances générales.
3. Ne paraphrase pas au-delà de ce qui est dans le contexte.
4. Si une définition exacte est disponible dans le contexte, cite-la fidèlement.
5. Si plusieurs extraits du contexte traitent du même sujet, synthétise-les.

FORMAT DE RÉPONSE :
- Commence directement par la réponse, sans introduction du type "Selon le contexte..."
- Pour une définition : commence par "[Terme] est / désigne / correspond à..."
- Pour une procédure : utilise une liste numérotée
- Sois concis : 3 à 8 phrases maximum sauf si plus de détails sont clairement demandés"""

        # ── Prompt utilisateur ────────────────────────────────────────────────
        # On sépare visuellement le contexte de la question pour que le modèle
        # ne confonde pas les deux.
        user_prompt = f"""### CONTEXTE DOCUMENTAIRE
{context}

### QUESTION
{query}

### RÉPONSE (basée exclusivement sur le contexte ci-dessus)"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": 1024,   # ← augmenté (600 tronquait les réponses)
                "top_p":       0.9,
                "repeat_penalty": 1.1, # ← réduit les répétitions
            }
        )

        return response['message']['content'].strip()