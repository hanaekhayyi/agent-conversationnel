"""
Agent de Classification Maroclear
Classifie les demandes des affilies en : Incident / Demande / Reclamation
Determine la priorite OLA (P1 a P4) et verifie si l'incident est fonde.

Herite de BaseAgent pour etre orchestrable avec RAGAgent.
Base sur le dispositif transitoire de gestion des incidents externes (23/01/2026).
"""

import re
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.message import AgentMessage, MessageType
from core.agent_context import AgentContext
from llm.ollama_client import OllamaClient


# =============================================================
# BASE DE CONNAISSANCES METIER MAROCLEAR
# Extraite du dispositif transitoire (23/01/2026)
# =============================================================

KNOWLEDGE_BASE = {

    "definitions": {
        "incident": (
            "Un INCIDENT est un dysfonctionnement technique ou fonctionnel imprévu "
            "qui empêche partiellement ou totalement l'utilisation normale d'un service "
            "ou système Maroclear. Il peut s'agir d'un blocage opérationnel, "
            "d'une indisponibilité d'application, ou d'un comportement anormal du système. "
            "L'incident est fondé s'il est réel et confirmé par le dispatcher, non fondé "
            "s'il résulte d'une erreur d'usage ou d'une mauvaise utilisation."
        ),
        "demande": (
            "Une DEMANDE est une sollicitation d'un affilié pour obtenir un service, "
            "une information, un document ou une assistance de la part de Maroclear. "
            "Elle ne correspond pas à un dysfonctionnement mais à un besoin planifié : "
            "demande de document, demande d'information, demande de retransmission de fichier, "
            "demande de réinitialisation de code PIN/token, demande d'accès ou d'habilitation."
        ),
        "reclamation": (
            "Une RÉCLAMATION est une expression d'insatisfaction formelle d'un affilié "
            "concernant un service reçu, un traitement effectué, une décision prise "
            "ou le comportement d'un collaborateur Maroclear. Elle implique un préjudice "
            "réel ou perçu et une demande de correction ou de réparation."
        ),
    },

    "services_systemes": {
        "Post-marché Bourse": {
            "systeme": "TCS BaNCS",
            "cas_incidents": [
                "Blocage général du marché / dénouement bloqué -> P1 Majeur",
                "Blocage d'un affilié (opérations impossibles) -> P2 Très haute",
                "Dysfonctionnement partiel contournable -> P3 Moyenne",
                "Bug affichage -> P4 Basse",
            ]
        },
        "Post-marché OTC": {
            "systeme": "TCS BaNCS",
            "cas_incidents": [
                "Blocage général / instructions bloquées / dénouement bloqué -> P1 Majeur",
                "Blocage d'un affilié (opérations impossibles) -> P2 Très haute",
                "Dysfonctionnement partiel contournable -> P3 Moyenne",
            ]
        },
        "Post-marché Repo's": {
            "systeme": "TCS BaNCS",
            "cas_incidents": [
                "Blocage général / instructions bloquées / dénouement bloqué -> P1 Majeur",
            ]
        },
        "OST (Opérations sur Titres)": {
            "systeme": "MyMaroclear",
            "cas_incidents": [
                "Impossibilité de prise en charge d'OST -> P2 Très haute",
                "Dysfonctionnement partiel de prise en charge d'OST -> P3 Moyenne",
            ]
        },
        "Référentiel Titres": {
            "systeme": "MyMaroclear",
            "cas_incidents": [
                "Impossibilité de prise en charge de l'admission/codification -> P2 Très haute",
                "Dysfonctionnement partiel de prise en charge d'un instrument -> P3 Moyenne",
            ]
        },
        "DSI - Exploitation": {
            "systeme": "Échange de fichiers",
        },
        "DSI - Applications": {
            "systeme": "Applications métier",
        },
        "DO - Post-marché & Reporting": {
            "systeme": "Reporting",
        },
        "DO - Référentiel": {
            "systeme": "Titres, Affiliés et OST",
        },
    },

    "priorites_ola": {
        "P1": {
            "label": "MAJEUR - Urgence marché",
            "definition": "Blocage général tout service confondu.",
            "prise_en_charge": "15 minutes",
            "resolution": "2 heures",
        },
        "P2": {
            "label": "TRÈS HAUTE - Bloquant",
            "definition": "Dysfonctionnement partiel empêchant le déroulement normal des processus.",
            "prise_en_charge": "30 minutes",
            "resolution": "4 heures",
        },
        "P3": {
            "label": "MOYENNE - Gênant",
            "definition": "Dysfonctionnement contournable ou sans impact majeur.",
            "prise_en_charge": "4 heures",
            "resolution": "J+1",
        },
        "P4": {
            "label": "BASSE - Mineur",
            "definition": "Bug cosmétique, anomalie mineure ne constituant aucun blocage.",
            "prise_en_charge": "8 heures",
            "resolution": "J+2",
        },
    },

    "circuit": {
        "0_declaration":  "Affilié déclare via MyMaroclear : objet, service, système, description, pièces jointes.",
        "1_reception":    "Dispatcher : réception, contrôle complétude, qualification fondé/non fondé, création ticket GLPI, affectation.",
        "2_traitement":   "Entité (DSI/DO) : prise en charge, diagnostic, actions correctives, mise à jour statuts GLPI.",
        "3_supervision":  "Contrôle Interne : validation SESAM (J) + rapport AMMC (J+5).",
        "4_qa":           "Quality Assurance : vérification complétude, respect SLA, identification écarts.",
        "5_reporting":    "Pilotage : extraction, confrontation MyMaroclear <-> GLPI, suivi KPI.",
    },

    "champs_glpi": [
        "Source de la demande = MyMaroclear (obligatoire)",
        "Catégorie = Système impacté",
        "Date de déclaration affilié (différente de la date de création du ticket GLPI)",
        "Description structurée : nom déclarant, affilié, ID incident, horodatage, objet, service, système, description",
        "Attribué à : technicien + groupe de traitement",
    ],

    "regles_qualification": [
        "Incident NON FONDE (erreur d'usage) -> réorienter, pas de ticket GLPI.",
        "Incident FONDE -> création OBLIGATOIRE d'un ticket GLPI, référence à conserver.",
        "Date déclaration affilié != Date création ticket GLPI (distinction obligatoire).",
        "Incidents avérés -> déclaration SESAM le jour J du signalement.",
        "Rapport AMMC -> J+5 du signalement.",
        "Dispatcher = rôle neutre : qualification + affectation uniquement.",
    ],
}


# =============================================================
# CONSTRUCTION DU PROMPT SYSTEME
# =============================================================

def _build_classification_system_prompt() -> str:
    """Construit le prompt système complet pour l'agent de classification."""

    kb = KNOWLEDGE_BASE

    priorites_text = "\n".join([
        f"  - {p} : {v['label']} | prise en charge {v['prise_en_charge']} | résolution {v['resolution']}"
        for p, v in kb["priorites_ola"].items()
    ])

    services_text = "\n".join([
        f"  - {svc} : système {info.get('systeme', '?')}"
        for svc, info in kb["services_systemes"].items()
    ])

    regles_text = "\n".join(
        f"  {i+1}. {r}" for i, r in enumerate(kb["regles_qualification"])
    )

    circuit_text = "\n".join(
        f"  Etape {k.split('_')[0]} -> {v}"
        for k, v in kb["circuit"].items()
    )

    return f"""Tu es l'Agent de Classification Maroclear, spécialisé dans la qualification des demandes des affiliés du portail MyMaroclear.

Ton rôle est d'analyser la situation décrite et de :
1. CLASSIFIER la demande : INCIDENT / DEMANDE / RECLAMATION
2. Si INCIDENT : déterminer la PRIORITE (P1 à P4) et le SERVICE concerné
3. VERIFIER si l'incident est FONDE ou NON FONDE
4. GUIDER l'affilié si des informations manquent (1 seule question à la fois)
5. EXPLIQUER le circuit de traitement applicable

DEFINITIONS OFFICIELLES MAROCLEAR
==================================
INCIDENT : {kb["definitions"]["incident"]}

DEMANDE : {kb["definitions"]["demande"]}

RECLAMATION : {kb["definitions"]["reclamation"]}

MATRICE DE PRIORITES OLA
========================
{priorites_text}

SERVICES ET SYSTEMES
====================
{services_text}

REGLES DE QUALIFICATION
=======================
{regles_text}

CIRCUIT DE TRAITEMENT
=====================
{circuit_text}

INSTRUCTIONS DE COMPORTEMENT
=============================
- Reponds TOUJOURS en français, de façon professionnelle et bienveillante.
- Si la classification est evidente, annonce-la directement avec justification.
- Si la situation est ambigue, pose UNE SEULE question ciblée.
- Pour un INCIDENT, toujours préciser : fonde/non fonde, priorité P1-P4, délais OLA, service et système, prochaines étapes.
- Pour une DEMANDE : indiquer la rubrique MyMaroclear à utiliser.
- Pour une RECLAMATION : valider le motif et décrire le circuit.
- Termine TOUJOURS par ce bloc de synthèse exact :

CLASSIFICATION : [INCIDENT / DEMANDE / RECLAMATION]
PRIORITE       : [P1 / P2 / P3 / P4 - si incident, sinon N/A]
SERVICE        : [service concerné]
SYSTEME        : [système impacté]
STATUT         : [Fondé / Non fondé / A vérifier]
PROCHAINE ETAPE: [action concrète recommandée]
"""


# =============================================================
# AGENT DE CLASSIFICATION
# =============================================================

class ClassificationAgent(BaseAgent):
    """
    Agent de classification des demandes Maroclear.
    Herite de BaseAgent pour s'integrer dans l'orchestrateur multi-agents.
    """

    # Mots-clés qui déclenchent cet agent
    INCIDENT_KEYWORDS = [
        "bloqué", "bloque", "inaccessible", "impossible", "erreur",
        "dysfonctionnement", "problème", "probleme", "panne", "bug",
        "ne fonctionne pas", "ne marche pas", "indisponible", "planté",
        "incident", "tcs bancs", "bancs", "mymaroclear ne",
    ]

    DEMANDE_KEYWORDS = [
        "demande", "besoin", "souhaite", "voudrais", "pouvez-vous",
        "pouvez vous", "retransmission", "réinitialisation", "reinitialisation",
        "pin", "token", "accès", "acces", "document", "relevé", "releve",
        "fichier", "formulaire", "créer un accès", "creer un acces",
    ]

    RECLAMATION_KEYWORDS = [
        "réclamation", "reclamation", "insatisfait", "inacceptable",
        "contestation", "conteste", "délai dépassé", "delai depasse",
        "toujours pas", "aucune réponse", "aucune reponse", "sans réponse",
        "préjudice", "prejudice", "erreur de traitement",
    ]

    def __init__(self, config: Dict[str, Any], llm_client: OllamaClient):
        super().__init__(agent_name="Classification_Agent", config=config)
        self.llm_client     = llm_client
        self.system_prompt  = _build_classification_system_prompt()
        self.log("Agent de Classification initialise")

    # =========================================================
    # ROUTING — decision de l'orchestrateur
    # =========================================================

    def can_handle(self, message: AgentMessage, context: AgentContext) -> bool:
        """
        Retourne True si le message ressemble à un incident, une demande
        ou une réclamation plutôt qu'à une question documentaire.

        Logique :
        - Presence de mots-clés de classification -> True
        - Message court et affirmatif (pas une question documentaire) -> True
        - Question commençant par "qu'est-ce", "comment", "quel" -> False
          (laissé au RAGAgent)
        """
        query_lower = message.content.lower()

        # Exclure explicitement les questions documentaires
        doc_starters = [
            "qu'est-ce", "qu est-ce", "c'est quoi", "c est quoi",
            "quelle est la définition", "définition de", "definition de",
            "comment fonctionne", "quel est le rôle", "quels sont les services",
            "comment devenir", "quelles sont les étapes",
        ]
        if any(query_lower.startswith(s) or s in query_lower for s in doc_starters):
            return False

        # Détecter les mots-clés métier de classification
        all_keywords = (
            self.INCIDENT_KEYWORDS
            + self.DEMANDE_KEYWORDS
            + self.RECLAMATION_KEYWORDS
        )
        return any(kw in query_lower for kw in all_keywords)

    # =========================================================
    # TRAITEMENT
    # =========================================================

    def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        """
        Traite le message et retourne la classification.
        Utilise l'historique de contexte pour les conversations multi-tours.
        """
        self.log(f"Classification de : {message.content[:80]}...")

        # Construire les messages avec historique si disponible
        messages = self._build_messages(message, context)

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )
            response_text = response.choices[0].message.content.strip()

        except Exception as e:
            self.log(f"Erreur LLM : {e}", level="ERROR")
            response_text = f"Erreur lors de la classification : {e}"

        # Extraire les résultats structurés
        classification = self._extract_field(response_text, "CLASSIFICATION")
        priorite       = self._extract_field(response_text, "PRIORITE")
        statut         = self._extract_field(response_text, "STATUT")
        service        = self._extract_field(response_text, "SERVICE")
        systeme        = self._extract_field(response_text, "SYSTEME")

        # Décider si le RAG est nécessaire en complément
        needs_rag = self._needs_rag(classification, message.content)

        return AgentMessage.create_response(
            sender=self.agent_name,
            content=response_text,
            metadata={
                "classification": classification,
                "priorite":       priorite,
                "statut":         statut,
                "service":        service,
                "systeme":        systeme,
                "needs_rag":      needs_rag,
                "needs_glpi":     (
                    classification == "INCIDENT"
                    and statut not in ("Non fondé", "")
                ),
            }
        )

    # =========================================================
    # HELPERS PRIVES
    # =========================================================

    def _build_messages(
        self,
        message: AgentMessage,
        context: AgentContext,
    ) -> List[Dict]:
        """
        Construit la liste de messages pour l'API LLM.
        Injecte l'historique de conversation depuis AgentContext.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Injecter les N derniers échanges pour le multi-tours
        history = context.get_history(last_n=6)
        for h in history:
            role = "user" if h.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": h["content"]})

        # Message courant
        messages.append({"role": "user", "content": message.content})
        return messages

    def _extract_field(self, text: str, field: str) -> str:
        """Extrait un champ du bloc de synthèse de la réponse."""
        pattern = rf"{field}\s*:\s*(.+)"
        match   = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().split("\n")[0].strip()
        return ""

    def _needs_rag(self, classification: str, content: str) -> bool:
        """
        Détermine si l'agent RAG doit être appelé en complément.
        Utile pour les demandes documentaires ou les questions procédurales.
        """
        if classification == "DEMANDE":
            doc_keywords = [
                "procédure", "procedure", "comment", "étapes", "etapes",
                "règle", "regle", "formulaire", "guide",
            ]
            return any(kw in content.lower() for kw in doc_keywords)
        return False

    # =========================================================
    # CAPACITES
    # =========================================================

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "agent_name":  self.agent_name,
            "description": "Classifie les demandes affiliés Maroclear en Incident/Demande/Réclamation",
            "handles": [
                "Déclaration d'incidents techniques (TCS BaNCS, MyMaroclear)",
                "Qualification de la priorité OLA (P1 à P4)",
                "Vérification fondé / non fondé",
                "Orientation vers le bon circuit de traitement",
                "Demandes de service (PIN, fichier, accès...)",
                "Réclamations formelles des affiliés",
            ],
            "routing_keywords": {
                "incident":    self.INCIDENT_KEYWORDS[:5],
                "demande":     self.DEMANDE_KEYWORDS[:5],
                "reclamation": self.RECLAMATION_KEYWORDS[:5],
            },
        }
