"""
Interface Streamlit — Maroclear Multi-Agent Assistant
Orchestre : ClassificationAgent + RAGAgent via OrchestratorAgent
Usage : streamlit run app.py
"""

import streamlit as st
import yaml
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from rag.embeddings import LocalEmbeddings
from rag.vector_store import LocalVectorStore
from rag.retriever import Retriever
from llm.ollama_client import OllamaClient
from agents.rag_agent import RAGAgent
from agents.classification_agent import ClassificationAgent
from agents.orchestrator_agent import OrchestratorAgent
from core.message import AgentMessage
from core.agent_context import AgentContext

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Maroclear · Assistant IA",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0d1b2a !important; }
.stApp { background-color: #f2f4f7; }

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #8090b0 !important; }

[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    border: 1px solid #1e3050 !important;
    color: #8090b0 !important;
    width: 100% !important;
    text-align: left !important;
    border-radius: 6px !important;
    margin-bottom: 4px !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #e8722a !important;
    color: #e8722a !important;
    background-color: #243650 !important;
}
[data-testid="stSidebar"] hr { border-color: #1a2e48 !important; }

.stFormSubmitButton > button {
    background-color: #e8722a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
}
.stFormSubmitButton > button:hover { background-color: #f0914d !important; }

.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1.5px solid #dde3ec !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #e8722a !important;
    box-shadow: 0 0 0 2px rgba(232,114,42,0.15) !important;
}

/* Messages utilisateur */
.msg-user {
    background-color: #1a2940;
    color: #e8edf5;
    border-radius: 14px 14px 2px 14px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 0.5rem 4rem;
    font-size: 0.9rem;
    line-height: 1.5;
}
.msg-user-label {
    font-size: 0.6rem;
    color: #f0914d;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* Messages assistant RAG */
.msg-bot {
    background-color: #ffffff;
    border: 1px solid #dde3ec;
    border-left: 4px solid #e8722a;
    border-radius: 2px 12px 12px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 4rem 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #1a2940;
}

/* Messages assistant Classification */
.msg-classif {
    background-color: #ffffff;
    border: 1px solid #dde3ec;
    border-left: 4px solid #5ab85c;
    border-radius: 2px 12px 12px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 4rem 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #1a2940;
}

/* Messages orchestrateur (les deux agents) */
.msg-both {
    background-color: #ffffff;
    border: 1px solid #dde3ec;
    border-left: 4px solid #7c5cbf;
    border-radius: 2px 12px 12px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 4rem 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #1a2940;
}

.msg-bot-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* Badge de classification */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 4px;
}
.badge-incident  { background: #fde8e8; color: #c0392b; }
.badge-demande   { background: #e8f4fd; color: #2980b9; }
.badge-reclam    { background: #f5e8fd; color: #8e44ad; }
.badge-p1 { background: #fde8e8; color: #c0392b; }
.badge-p2 { background: #fef3e8; color: #e67e22; }
.badge-p3 { background: #fefde8; color: #d4ac0d; }
.badge-p4 { background: #eafde8; color: #27ae60; }
.badge-rag { background: #e8f0fe; color: #2c5ab5; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INIT & CACHE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_resource
def init_orchestrator(_config):
    """Initialise tous les agents et retourne l'orchestrateur."""

    vs  = LocalVectorStore(
        persist_directory=_config["vector_store"]["persist_directory"],
        collection_name=_config["vector_store"]["collection_name"],
    )
    emb = LocalEmbeddings(_config["embeddings"]["model"])
    ret = Retriever(
        vector_store=vs,
        embeddings=emb,
        top_k=_config["rag"].get("top_k", 8),
        similarity_threshold=_config["rag"].get("similarity_threshold", 0.4),
    )
    llm = OllamaClient(
        model=_config["llm"]["model"],
        temperature=_config["llm"].get("temperature", 0.1),
        api_key=_config["llm"].get("api_key"),
        base_url=_config["llm"].get("base_url", "https://openrouter.ai/api/v1"),
        max_tokens=_config["llm"].get("max_tokens", 1024),
    )

    rag_agent = RAGAgent(
        config=_config,
        retriever=ret,
        llm_client=llm,
    )

    classification_agent = ClassificationAgent(
        config=_config,
        llm_client=llm,
    )

    orchestrator = OrchestratorAgent(
        config=_config,
        classification_agent=classification_agent,
        rag_agent=rag_agent,
    )

    stats = vs.get_collection_stats()
    return orchestrator, stats

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for k, v in [
    ("messages",  []),
    ("q_count",   0),
    ("pending_q", ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v

config = load_config()

try:
    orchestrator, stats = init_orchestrator(config)
    system_ready = True
    doc_count    = stats.get("count", 0)
except Exception as e:
    system_ready = False
    doc_count    = 0
    st.error(f"Erreur d'initialisation : {e}")

# ══════════════════════════════════════════════════════════════════════════════
# EXEMPLES PAR CATEGORIE
# ══════════════════════════════════════════════════════════════════════════════

EXEMPLES_INCIDENTS = [
    "TCS BaNCS est inaccessible, toutes mes operations de bourse sont bloquees.",
    "Je ne peux pas soumettre mes OST sur MyMaroclear depuis ce matin.",
    "Le denouement de mes instructions OTC est bloque depuis 9h.",
    "Je constate un bug d'affichage sur la page des cours.",
]

EXEMPLES_DEMANDES = [
    "Pouvez-vous me retransmettre le fichier de reglement du 05/03/2026 ?",
    "J'ai oublie mon code PIN MyMaroclear, pouvez-vous le reinitialiser ?",
    "Demande de creation d'un acces MyMaroclear pour un nouveau collaborateur.",
]

EXEMPLES_RECLAMATIONS = [
    "Mon incident declare le 01/03 n'a toujours pas ete traite apres 5 jours.",
    "Le traitement de mon dossier OST a ete effectue avec une erreur que je conteste.",
]

EXEMPLES_QUESTIONS = [
    "C'est quoi Maroclear ?",
    "Comment devenir affilie ?",
    "Qu'est-ce qu'un OPCVM ?",
    "Quel est le role du depositaire central ?",
    "Quels sont les services aux affilies ?",
]

model_short = config["llm"]["model"].split("/")[-1][:24]
emb_short   = config["embeddings"]["model"][:20]
now         = datetime.now().strftime("%d %b %Y · %H:%M")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("## MAROCLEAR")
    st.caption("Assistant IA · Multi-Agents v2.0")
    st.divider()

    # Statut systeme
    st.markdown("**STATUT SYSTEME**")
    ok = "OK" if system_ready else "ERR"
    st.markdown(f"[{ok}] **Base vectorielle** — {doc_count} chunks")
    st.markdown(f"[{ok}] **RAGAgent** — Questions documentaires")
    st.markdown(f"[{ok}] **ClassificationAgent** — Incidents / Demandes / Reclamations")
    st.markdown(f"[{ok}] **OrchestratorAgent** — Routage automatique")
    st.markdown(f"[{ok}] **LLM** — {model_short}")
    st.divider()

    # Actions
    st.markdown("**ACTIONS**")
    if st.button("Effacer la conversation", key="clear_btn"):
        st.session_state.messages  = []
        st.session_state.q_count   = 0
        st.session_state.pending_q = ""
        if system_ready:
            orchestrator.reset_context()
        st.rerun()

    st.divider()

    # Exemples Incidents
    st.markdown("**INCIDENTS**")
    for i, q in enumerate(EXEMPLES_INCIDENTS):
        label = q[:45] + "..." if len(q) > 45 else q
        if st.button(label, key=f"inc_{i}"):
            st.session_state.pending_q = q
            st.rerun()

    st.divider()

    # Exemples Demandes
    st.markdown("**DEMANDES**")
    for i, q in enumerate(EXEMPLES_DEMANDES):
        label = q[:45] + "..." if len(q) > 45 else q
        if st.button(label, key=f"dem_{i}"):
            st.session_state.pending_q = q
            st.rerun()

    st.divider()

    # Exemples Reclamations
    st.markdown("**RECLAMATIONS**")
    for i, q in enumerate(EXEMPLES_RECLAMATIONS):
        label = q[:45] + "..." if len(q) > 45 else q
        if st.button(label, key=f"rec_{i}"):
            st.session_state.pending_q = q
            st.rerun()

    st.divider()

    # Exemples Questions documentaires
    st.markdown("**QUESTIONS DOCUMENTAIRES**")
    for i, q in enumerate(EXEMPLES_QUESTIONS):
        if st.button(q, key=f"qdoc_{i}"):
            st.session_state.pending_q = q
            st.rerun()

    st.divider()
    st.caption("2026. Maroclear. Tous droits reserves.")

# ══════════════════════════════════════════════════════════════════════════════
# TRAITEMENT DU MESSAGE (sidebar ou formulaire)
# ══════════════════════════════════════════════════════════════════════════════

def process_question(question: str):
    """Envoie la question a l'orchestrateur et stocke la reponse."""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.q_count += 1

    with st.spinner("Analyse en cours..."):
        try:
            result = orchestrator.chat(question)
            st.session_state.messages.append({
                "role":           "assistant",
                "content":        result["response"],
                "agent_used":     result["agent_used"],
                "route":          result["route"],
                "classification": result["classification"],
                "priorite":       result["priorite"],
                "statut":         result["statut"],
                "service":        result["service"],
                "systeme":        result["systeme"],
                "needs_glpi":     result["needs_glpi"],
                "rag_sources":    result["rag_sources"],
            })
        except Exception as e:
            st.session_state.messages.append({
                "role":       "assistant",
                "content":    f"Erreur : {e}",
                "agent_used": "error",
                "route":      "error",
            })


if st.session_state.pending_q:
    question = st.session_state.pending_q
    st.session_state.pending_q = ""
    process_question(question)
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# EN-TETE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:0.3rem 0;">
        <div style="background:#e8722a;width:42px;height:42px;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.3rem;flex-shrink:0;">
            &#127963;
        </div>
        <div>
            <div style="font-size:1.3rem;font-weight:700;line-height:1.2;">
                <span style="color:#0d1b2a;">MAROC</span>
                <span style="color:#e8722a;">CLEAR</span>
            </div>
            <div style="font-size:0.75rem;color:#6b7a94;margin-top:2px;">
                Assistant Multi-Agents · Classification + RAG · Depositaire Central des Titres
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(
        f"<div style='text-align:right;color:#6b7a94;font-size:0.8rem;"
        f"padding-top:0.5rem;'>{now}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ZONE CHAT
# ══════════════════════════════════════════════════════════════════════════════

def get_agent_label(agent_used: str, route: str) -> tuple:
    """
    Retourne (label_texte, couleur_label, classe_css_bulle)
    selon l'agent qui a repondu.
    """
    if route == "rag":
        return "RAGAgent — Base documentaire", "#5ab85c", "msg-bot"
    elif route == "classification":
        return "ClassificationAgent — Qualification", "#e8722a", "msg-classif"
    elif "+" in route:
        return "ClassificationAgent + RAGAgent", "#7c5cbf", "msg-both"
    else:
        return "Assistant Maroclear", "#5ab85c", "msg-bot"


def build_badges(msg: dict) -> str:
    """Construit les badges HTML de classification."""
    badges = ""

    classif = msg.get("classification", "")
    if classif == "INCIDENT":
        badges += '<span class="badge badge-incident">INCIDENT</span>'
    elif classif == "DEMANDE":
        badges += '<span class="badge badge-demande">DEMANDE</span>'
    elif classif in ("RECLAMATION", "RÉCLAMATION"):
        badges += '<span class="badge badge-reclam">RECLAMATION</span>'

    priorite = msg.get("priorite", "")
    if priorite in ("P1", "P2", "P3", "P4"):
        p = priorite.lower()
        badges += f'<span class="badge badge-{p}">{priorite}</span>'

    sources = msg.get("rag_sources", [])
    if sources:
        badges += '<span class="badge badge-rag">Sources RAG</span>'

    return badges


if not st.session_state.messages:
    st.info(
        "Utilisez le menu lateral pour selectionner un exemple, "
        "ou tapez directement votre question ci-dessous.\n\n"
        "L'orchestrateur redirigera automatiquement votre message vers "
        "le bon agent (Classification ou RAG)."
    )
else:
    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">'
                f'<div class="msg-user-label">Votre message</div>'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:
            route      = msg.get("route", "rag")
            agent_used = msg.get("agent_used", "")
            label, color, css_class = get_agent_label(agent_used, route)

            badges    = build_badges(msg)
            content   = msg["content"].replace("\n", "<br>")

            # Ligne sources RAG
            sources   = msg.get("rag_sources", [])
            src_text  = ""
            if sources:
                names    = [s.split("/")[-1].replace("_", " ") for s in sources]
                src_text = (
                    f'<div style="margin-top:0.6rem;padding-top:0.5rem;'
                    f'border-top:1px solid #edf0f5;font-size:0.72rem;color:#8090b0;">'
                    f'Sources : {" · ".join(names)}</div>'
                )

            # Ligne ticket GLPI
            glpi_text = ""
            if msg.get("needs_glpi"):
                glpi_text = (
                    f'<div style="margin-top:0.6rem;padding:0.4rem 0.8rem;'
                    f'background:#fef3e8;border-radius:6px;'
                    f'font-size:0.75rem;color:#e67e22;font-weight:600;">'
                    f'Un ticket GLPI doit etre cree pour cet incident.'
                    f'</div>'
                )

            # Infos de classification supplementaires
            classif_details = ""
            service = msg.get("service", "")
            systeme = msg.get("systeme", "")
            statut  = msg.get("statut", "")
            if service or systeme or statut:
                parts = []
                if service: parts.append(f"Service : {service}")
                if systeme: parts.append(f"Systeme : {systeme}")
                if statut:  parts.append(f"Statut : {statut}")
                classif_details = (
                    f'<div style="margin-top:0.5rem;font-size:0.75rem;color:#6b7a94;">'
                    + " &nbsp;|&nbsp; ".join(parts)
                    + '</div>'
                )

            st.markdown(
                f'<div class="{css_class}">'
                f'<div class="msg-bot-label" style="color:{color};">'
                f'{label}'
                f'</div>'
                f'{badges}'
                f'{"<br>" if badges else ""}'
                f'{content}'
                f'{classif_details}'
                f'{src_text}'
                f'{glpi_text}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# FORMULAIRE D'INPUT
# ══════════════════════════════════════════════════════════════════════════════

st.divider()

if st.session_state.q_count > 0:
    s = "s" if st.session_state.q_count > 1 else ""
    st.caption(f"{st.session_state.q_count} message{s} dans cette session")

with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([6, 1])
    with col_in:
        user_input = st.text_input(
            "message",
            placeholder="Decrivez votre incident, demande, reclamation ou posez une question...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Envoyer", use_container_width=True)

if submitted and user_input and user_input.strip() and system_ready:
    process_question(user_input.strip())
    st.rerun()

elif submitted and not system_ready:
    st.error("Systeme non initialise. Verifiez config.yaml et relancez.")