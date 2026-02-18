"""
Client pour Ollama en local
"""

import ollama
from typing import Dict, Any

class OllamaClient:
    """Client Ollama local"""
    
    def __init__(self, model: str = "mistral:7b-instruct", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        
        # Vérifier qu'Ollama est disponible
        try:
            ollama.list()
            print(f"Ollama connecté - Modèle: {model}")
        except Exception as e:
            print(f"Erreur Ollama: {e}")
            raise
    
    def generate_rag_response(self, query: str, context: str) -> str:
            """Génère une réponse basée sur le contexte fourni"""
            
            system_prompt = """Tu es un assistant expert sur Maroclear, le dépositaire central des titres au Maroc.

        RÈGLES IMPORTANTES :
        1. Réponds UNIQUEMENT en te basant sur le CONTEXTE fourni ci-dessous
        2. Si l'information n'est pas dans le contexte, dis clairement "Cette information n'est pas disponible dans ma base de connaissances"
        3. Sois précis, professionnel et pédagogue
        4. Structure ta réponse de manière claire (utilise des listes si pertinent)
        5. Si le contexte contient une définition du glossaire, cite-la exactement
        6. Évite les généralités - donne des détails concrets du contexte
        7. Si plusieurs sources donnent des infos complémentaires, synthétise-les

        FORMAT DE RÉPONSE :
        - Commence directement par la réponse (pas de "Selon le contexte fourni...")
        - Sois concis mais complet
        - Si c'est une définition, commence par "X est..."
        """
            
            # 🔧 AMÉLIORATION : Mieux structurer le contexte
            user_prompt = f"""CONTEXTE DOCUMENTAIRE :
        {context}

        ---

        QUESTION DE L'UTILISATEUR :
        {query}

        ---

        RÉPONSE (basée uniquement sur le contexte ci-dessus) :"""
            
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": 600  # 🔧 Augmenté de 500 à 600
                }
            )
            
            return response['message']['content'].strip()