"""
Agent RAG principal - Répond aux questions métier sur Maroclear
"""

from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.message import AgentMessage, MessageType
from core.agent_context import AgentContext
from rag.retriever import Retriever
from llm.ollama_client import OllamaClient


class RAGAgent(BaseAgent):
    """Agent de questions/réponses basé sur RAG"""

    def __init__(self, config: Dict[str, Any], retriever: Retriever, llm_client: OllamaClient):
        super().__init__(agent_name="RAG_Agent", config=config)
        self.retriever  = retriever
        self.llm_client = llm_client
        self.log("Agent RAG initialisé")

    # =========================================================
    # ROUTING
    # =========================================================

    def can_handle(self, message: AgentMessage, context: AgentContext) -> bool:
        query_lower = message.content.lower()

        rag_keywords = [
            "c'est quoi", "c est quoi", "qu'est-ce", "qu est-ce",
            "quelle est", "quel est", "quels sont",
            "définition", "definition", "expliquer", "signifie", "veut dire",
            "comment", "pourquoi", "role", "rôle", "mission",
            "maroclear", "dépositaire", "depositaire", "affilié", "affilie",
            "bourse", "titre", "opcvm", "service", "post-marché",
            "dénouement", "denouement", "conservation", "règlement", "reglement"
        ]
        question_starters = [
            "qui", "que", "quoi", "comment", "pourquoi",
            "où", "ou", "quand", "quel"
        ]

        return (
            any(kw in query_lower for kw in rag_keywords)
            or any(query_lower.startswith(s) for s in question_starters)
        )

    # =========================================================
    # TRAITEMENT
    # =========================================================

    def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        self.log(f"Traitement : {message.content[:100]}...")

        retrieved_docs = self.retriever.retrieve(message.content)

        if not retrieved_docs:
            return AgentMessage.create_response(
                sender=self.agent_name,
                content=(
                    "Je n'ai pas trouvé d'informations pertinentes "
                    "dans la base de connaissances Maroclear. "
                    "Pouvez-vous reformuler votre question ?"
                ),
                metadata={"retrieved_docs_count": 0, "sources": []}
            )

        context_text = self._build_context(retrieved_docs)

        response_text = self.llm_client.generate_rag_response(
            query=message.content,
            context=context_text,
        )

        sources = list(dict.fromkeys(
            doc['metadata'].get('source', 'unknown') for doc in retrieved_docs
        ))

        return AgentMessage.create_response(
            sender=self.agent_name,
            content=response_text,
            metadata={
                "retrieved_docs_count": len(retrieved_docs),
                "sources": sources,
            }
        )

    # =========================================================
    # CONSTRUCTION DU CONTEXTE
    # =========================================================

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Construit le contexte transmis au LLM.

        Améliorations par rapport à l'ancienne version :
        - Les entrées du glossaire sont placées EN PREMIER (définitions exactes)
        - Chaque extrait est clairement labellisé avec son type et son score
        - Les doublons de contenu sont éliminés
        """
        # Séparer glossaire et documents PDF
        glossaire_docs = [d for d in retrieved_docs if d['metadata'].get('type') == 'glossaire']
        other_docs     = [d for d in retrieved_docs if d['metadata'].get('type') != 'glossaire']

        parts = []

        # ── Définitions du glossaire en tête ─────────────────────────────────
        if glossaire_docs:
            parts.append("=== DÉFINITIONS DU GLOSSAIRE ===")
            seen = set()
            for doc in glossaire_docs:
                content = doc['content'].strip()
                if content in seen:
                    continue
                seen.add(content)
                term = doc['metadata'].get('term', '')
                parts.append(f"• {content}")

        # ── Extraits des documents PDF ────────────────────────────────────────
        if other_docs:
            parts.append("\n=== EXTRAITS DES DOCUMENTS ===")
            seen = set()
            for i, doc in enumerate(other_docs, 1):
                content = doc['content'].strip()
                if content in seen:
                    continue
                seen.add(content)
                source = doc['metadata'].get('source', 'Document')
                score  = doc.get('score', 0)
                parts.append(f"[Extrait {i} — {source} — pertinence: {score:.2f}]\n{content}")

        result = "\n\n".join(parts)
        print(f"\n{'='*40}\nCONTEXTE ENVOYÉ AU LLM:\n{result}\n{'='*40}\n")  # DEBUG
        return result


    # =========================================================
    # CAPACITÉS
    # =========================================================

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "agent_name":  self.agent_name,
            "description": "Répond aux questions métier sur Maroclear",
            "handles": [
                "Questions sur Maroclear (activités, services)",
                "Procédure d'affiliation",
                "Définitions du glossaire",
                "Questions générales métier",
            ],
            "knowledge_base_stats": self.retriever.vector_store.get_collection_stats(),
        }