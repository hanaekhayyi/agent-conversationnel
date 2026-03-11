"""
Agent Orchestrateur Maroclear
Recoit tous les messages des affilies et decide quel agent appeler.

Logique de routage :
  1. ClassificationAgent -> si le message ressemble a un incident / demande / reclamation
  2. RAGAgent            -> si le message est une question documentaire metier
  3. Les deux            -> si la classification detecte needs_rag = True
  4. Reponse par defaut  -> si aucun agent ne peut traiter

S'appuie sur can_handle() de chaque agent pour le routage automatique.
"""

from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from agents.rag_agent import RAGAgent
from agents.classification_agent import ClassificationAgent
from core.message import AgentMessage, MessageType
from core.agent_context import AgentContext


# =============================================================
# AGENT ORCHESTRATEUR
# =============================================================

class OrchestratorAgent(BaseAgent):
    """
    Orchestrateur central du systeme multi-agents Maroclear.

    Responsabilites :
    - Recevoir le message de l'affilie
    - Decider quel(s) agent(s) appeler via can_handle()
    - Combiner les reponses si plusieurs agents sont sollicites
    - Maintenir le contexte de conversation entre les tours
    """

    DEFAULT_RESPONSE = (
        "Je n'ai pas pu identifier le type de votre demande. "
        "Pourriez-vous preciser s'il s'agit :\n"
        "- D'un incident technique (dysfonctionnement, blocage)\n"
        "- D'une demande de service (document, acces, information)\n"
        "- D'une reclamation (insatisfaction, litige)\n"
        "- D'une question sur les services ou procedures Maroclear"
    )

    def __init__(
        self,
        config: Dict[str, Any],
        classification_agent: ClassificationAgent,
        rag_agent: RAGAgent,
    ):
        super().__init__(agent_name="Orchestrator_Agent", config=config)
        self.classification_agent = classification_agent
        self.rag_agent            = rag_agent
        self.context              = AgentContext()

        self.log("Orchestrateur initialise")
        self.log(f"Agents enregistres : ClassificationAgent, RAGAgent")

    # =========================================================
    # ROUTING
    # =========================================================

    def can_handle(self, message: AgentMessage, context: AgentContext) -> bool:
        """L'orchestrateur accepte tous les messages."""
        return True

    def _decide_route(self, message: AgentMessage) -> str:
        """
        Determine la route a emprunter.

        Retourne :
          "classification" -> ClassificationAgent uniquement
          "rag"            -> RAGAgent uniquement
          "both"           -> Classification en premier, RAG en complement
          "none"           -> Aucun agent ne peut traiter
        """
        can_classif = self.classification_agent.can_handle(message, self.context)
        can_rag     = self.rag_agent.can_handle(message, self.context)

        self.log(f"Routage -> ClassificationAgent={can_classif} | RAGAgent={can_rag}")

        if can_classif and can_rag:
            # Les deux sont eligibles : classification en priorite
            # Le RAG completera si needs_rag=True dans les metadonnees
            return "classification"
        elif can_classif:
            return "classification"
        elif can_rag:
            return "rag"
        else:
            return "none"

    # =========================================================
    # TRAITEMENT PRINCIPAL
    # =========================================================

    def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        """
        Point d'entree principal.
        Appele par la boucle de conversation pour chaque message utilisateur.
        """
        self.log(f"Message recu : {message.content[:80]}...")

        # Sauvegarder le message utilisateur dans le contexte
        self.context.add_message(
            sender="user",
            content=message.content,
            role="user"
        )

        route = self._decide_route(message)
        self.log(f"Route choisie : {route}")

        if route == "classification":
            response = self._handle_classification(message)

        elif route == "rag":
            response = self._handle_rag(message)

        elif route == "both":
            response = self._handle_both(message)

        else:
            response = AgentMessage.create_response(
                sender=self.agent_name,
                content=self.DEFAULT_RESPONSE,
                metadata={"route": "none", "agent_used": "orchestrator"}
            )

        # Sauvegarder la reponse dans le contexte
        self.context.add_message(
            sender=response.sender_agent,
            content=response.content,
            role="assistant"
        )

        return response

    # =========================================================
    # HANDLERS PAR ROUTE
    # =========================================================

    def _handle_classification(self, message: AgentMessage) -> AgentMessage:
        """
        Appelle ClassificationAgent.
        Si la reponse contient needs_rag=True, appelle aussi le RAGAgent
        et fusionne les deux reponses.
        """
        self.log("Delegation -> ClassificationAgent")
        classif_response = self.classification_agent.process(message, self.context)

        # Verifier si le RAG doit completer la reponse
        needs_rag = classif_response.metadata.get("needs_rag", False)

        if needs_rag and self.rag_agent.can_handle(message, self.context):
            self.log("Classification -> needs_rag=True, complement RAGAgent")
            rag_response = self.rag_agent.process(message, self.context)
            return self._merge_responses(classif_response, rag_response)

        classif_response.metadata["route"]      = "classification"
        classif_response.metadata["agent_used"] = "ClassificationAgent"
        return classif_response

    def _handle_rag(self, message: AgentMessage) -> AgentMessage:
        """Appelle RAGAgent pour les questions documentaires."""
        self.log("Delegation -> RAGAgent")
        rag_response = self.rag_agent.process(message, self.context)
        rag_response.metadata["route"]      = "rag"
        rag_response.metadata["agent_used"] = "RAGAgent"
        return rag_response

    def _handle_both(self, message: AgentMessage) -> AgentMessage:
        """
        Appelle les deux agents et fusionne les reponses.
        Classification en premier, RAG en complement.
        """
        self.log("Delegation -> ClassificationAgent + RAGAgent")
        classif_response = self.classification_agent.process(message, self.context)
        rag_response     = self.rag_agent.process(message, self.context)
        return self._merge_responses(classif_response, rag_response)

    # =========================================================
    # FUSION DES REPONSES
    # =========================================================

    def _merge_responses(
        self,
        classif_response: AgentMessage,
        rag_response: AgentMessage,
    ) -> AgentMessage:
        """
        Fusionne la reponse de classification et la reponse RAG.
        La classification reste en tete, le RAG apporte le complement documentaire.
        """
        merged_content = (
            classif_response.content
            + "\n\n"
            + "=" * 50
            + "\n"
            + "INFORMATIONS COMPLEMENTAIRES (Base documentaire Maroclear)"
            + "\n"
            + "=" * 50
            + "\n\n"
            + rag_response.content
        )

        merged_metadata = {
            **classif_response.metadata,
            "rag_sources":  rag_response.metadata.get("sources", []),
            "rag_docs":     rag_response.metadata.get("retrieved_docs_count", 0),
            "route":        "classification+rag",
            "agent_used":   "ClassificationAgent + RAGAgent",
        }

        return AgentMessage.create_response(
            sender=self.agent_name,
            content=merged_content,
            metadata=merged_metadata,
        )

    # =========================================================
    # BOUCLE DE CONVERSATION
    # =========================================================

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Interface simplifiee pour la boucle de conversation.
        Retourne un dict avec la reponse et les metadonnees utiles.

        Usage :
            orchestrator = OrchestratorAgent(config, classif_agent, rag_agent)
            result = orchestrator.chat("TCS BaNCS est inaccessible")
            print(result["response"])
            print(result["agent_used"])
            print(result["classification"])
        """
        message = AgentMessage.create_query(
            sender="user",
            content=user_input,
        )

        response = self.process(message, self.context)

        return {
            "response":       response.content,
            "agent_used":     response.metadata.get("agent_used", "unknown"),
            "route":          response.metadata.get("route", "unknown"),
            "classification": response.metadata.get("classification", ""),
            "priorite":       response.metadata.get("priorite", ""),
            "statut":         response.metadata.get("statut", ""),
            "service":        response.metadata.get("service", ""),
            "systeme":        response.metadata.get("systeme", ""),
            "needs_glpi":     response.metadata.get("needs_glpi", False),
            "rag_sources":    response.metadata.get("rag_sources", []),
        }

    def reset_context(self):
        """Reinitialise le contexte (nouvelle session)."""
        self.context = AgentContext()
        self.log("Contexte reinitialise")

    # =========================================================
    # CAPACITES
    # =========================================================

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "agent_name":  self.agent_name,
            "description": "Orchestrateur central du systeme multi-agents Maroclear",
            "routes": {
                "classification": "Incidents, demandes, reclamations des affilies",
                "rag":            "Questions documentaires et metier sur Maroclear",
                "classification+rag": "Demandes necessitant classification ET documentation",
            },
            "agents": {
                "ClassificationAgent": self.classification_agent.get_capabilities(),
                "RAGAgent":            self.rag_agent.get_capabilities(),
            },
        }
