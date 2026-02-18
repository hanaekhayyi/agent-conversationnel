"""
Agent RAG principal - Répond aux questions métier sur Maroclear
"""

import yaml
from typing import Dict, Any
from agents.base_agent import BaseAgent
from core.message import AgentMessage, MessageType
from core.agent_context import AgentContext
from rag.retriever import Retriever
from llm.ollama_client import OllamaClient

class RAGAgent(BaseAgent):
    """Agent de questions/réponses basé sur RAG"""
    
    def __init__(self, config: Dict[str, Any], retriever: Retriever, llm_client: OllamaClient):
        super().__init__(agent_name="RAG_Agent", config=config)
        
        self.retriever = retriever
        self.llm_client = llm_client
        
        self.log("Agent RAG initialisé")
    
    def can_handle(self, message: AgentMessage, context: AgentContext) -> bool:
        """
        Le RAG Agent gère :
        - Questions sur Maroclear (activités, services, affiliation)
        - Demandes de définitions (glossaire)
        - Questions générales métier
        """
        
        query_lower = message.content.lower()
        
        # 🔧 AMÉLIORATION : Keywords plus permissifs
        rag_keywords = [
            # Questions
            "c'est quoi", "c est quoi", "qu'est-ce", "qu est-ce", "quest-ce",
            "quelle est", "quel est", "quels sont",
            
            # Définitions
            "définition", "definition", "expliquer", "expliquez",
            "signifie", "veut dire",
            
            # Informations
            "comment", "pourquoi", "role", "rôle", "mission",
            
            # Termes métier Maroclear
            "maroclear", "dépositaire", "depositaire", "affilié", "affilie",
            "bourse", "titre", "opcvm", "service", "post-marché", "post-marche",
            "dénouement", "denouement", "conservation", "règlement", "reglement"
        ]
        
        # Vérifier si au moins un keyword est présent
        has_keyword = any(keyword in query_lower for keyword in rag_keywords)
        
        # 🆕 AJOUT : Si la question commence par un mot interrogatif, accepter
        question_starters = ["qui", "que", "quoi", "comment", "pourquoi", "où", "ou", "quand", "quel"]
        starts_with_question = any(query_lower.startswith(starter) for starter in question_starters)
        
        return has_keyword or starts_with_question
    def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        """Traite une question métier"""
        
        self.log(f"Traitement de la question: {message.content[:100]}...")
        
        # 1. Retrieve - Rechercher documents pertinents
        retrieved_docs = self.retriever.retrieve(message.content)
        
        if not retrieved_docs:
            return AgentMessage.create_response(
                sender=self.agent_name,
                content="Je n'ai pas trouvé d'informations pertinentes dans ma base de connaissances sur Maroclear. Pouvez-vous reformuler votre question ?",
                metadata={"retrieved_docs_count": 0}
            )
        
        # 2. Generate - Générer la réponse avec le LLM
        context_text = self._build_context(retrieved_docs)
        
        response_text = self.llm_client.generate_rag_response(
            query=message.content,
            context=context_text
        )
        
        # 3. Return response
        return AgentMessage.create_response(
            sender=self.agent_name,
            content=response_text,
            metadata={
                "retrieved_docs_count": len(retrieved_docs),
                "sources": [doc['metadata'].get('source', 'unknown') for doc in retrieved_docs]
            }
        )
    
    def _build_context(self, retrieved_docs: list) -> str:
        """Construit le contexte à partir des documents récupérés"""
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Document')
            content = doc['content']
            score = doc.get('score', 0)
            
            context_parts.append(
                f"[Source {i}: {source} (pertinence: {score:.2f})]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Capacités de l'agent RAG"""
        return {
            "agent_name": self.agent_name,
            "description": "Répond aux questions métier sur Maroclear",
            "handles": [
                "Questions sur Maroclear (activités, services)",
                "Procédure d'affiliation",
                "Définitions du glossaire",
                "Questions générales métier"
            ],
            "knowledge_base_stats": self.retriever.vector_store.get_collection_stats()
        }