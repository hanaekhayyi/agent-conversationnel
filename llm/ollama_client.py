"""
Client LLM via OpenRouter API (compatible OpenAI)
Remplace l'ancien client Ollama local.

Installation :
    pip install openai

Usage dans config.yaml :
    llm:
      provider: "openrouter"
      model: "arcee-ai/trinity-large-preview:free"
      base_url: "https://openrouter.ai/api/v1"
      api_key: "sk-or-..."
      temperature: 0.1
      max_tokens: 1024
"""

from openai import OpenAI
from typing import Dict, Any


class OllamaClient:
    """
    Client OpenRouter — conserve le nom OllamaClient pour ne pas modifier
    les imports dans main.py et rag_agent.py.
    """

    def __init__(
        self,
        model: str         = "arcee-ai/trinity-large-preview:free",
        temperature: float = 0.1,
        api_key: str       = None,
        base_url: str      = "https://openrouter.ai/api/v1",
        max_tokens: int    = 1024,
    ):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

        if not api_key:
            raise ValueError(
                "api_key manquant. "
                "Ajoutez llm.api_key dans config.yaml ou passez-le en argument."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        print(f"OpenRouter connecté — Modèle : {model}")

    def generate_rag_response(self, query: str, context: str) -> str:
        """Génère une réponse basée sur le contexte fourni."""

        system_prompt = """Tu es un assistant spécialisé sur Maroclear, le dépositaire central des titres au Maroc.

RÈGLES ABSOLUES — tu dois les respecter sans exception :
1. Réponds UNIQUEMENT en utilisant les informations contenues dans le CONTEXTE fourni.
2. Si l'information demandée n'est PAS dans le contexte, réponds exactement :
   "Je n'ai pas cette information dans les documents disponibles."
   Ne tente JAMAIS de compléter avec tes connaissances générales.
3. Ne paraphrase pas au-delà de ce qui est dans le contexte.
4. Si une définition exacte est disponible dans le contexte, cite-la fidèlement.
5. Si plusieurs extraits du contexte traitent du même sujet, synthétise-les.

FORMAT DE RÉPONSE — choisis automatiquement selon le contenu :

CAS 1 — DÉFINITION (question du type "c'est quoi", "qu'est-ce que", "définition de") :
  → Réponds en 1 à 3 phrases, en commençant par "[Terme] est / désigne / correspond à..."
  → Pas de liste, pas de titre, juste un paragraphe fluide.

CAS 2 — PROCÉDURE / ÉTAPES (question du type "comment", "quelles sont les étapes", "comment devenir", "comment faire") :
  → Utilise une liste numérotée claire :
     1. Première étape
     2. Deuxième étape
     ...
  → Ajoute un titre court avant la liste si utile (ex: "**Pour devenir affilié chez Maroclear :**")
  → Si certaines étapes ont des sous-conditions, utilise des tirets (–) en dessous.

CAS 3 — LISTE DE SERVICES / CARACTÉRISTIQUES (question du type "quels sont", "quelles sont") :
  → Utilise une liste à puces (•) :
     • Premier élément
     • Deuxième élément
  → Ajoute une phrase d'introduction avant la liste.

CAS 4 — QUESTION GÉNÉRALE / RÔLE / MISSION :
  → Réponds en 2 à 4 phrases organisées en paragraphe.
  → Pas de liste sauf si le contexte en contient une explicitement.

RÈGLE UNIVERSELLE : ne jamais mélanger les formats. Si la réponse est une définition, pas de liste. Si c'est une procédure, pas de paragraphe continu."""

        user_prompt = f"""### CONTEXTE DOCUMENTAIRE
{context}

### QUESTION
{query}

### RÉPONSE (basée exclusivement sur le contexte ci-dessus)"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[ERREUR OpenRouter] {e}")
            return f"Erreur lors de la génération de la réponse : {e}"