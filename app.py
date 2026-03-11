"""
Interface Streamlit — Maroclear RAG Assistant
Sidebar 100% native Streamlit, CSS minimal et fiable.
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
# CSS MINIMAL — uniquement ce qui est nécessaire et fiable
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Couleur de fond sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1b2a !important;
}

/* Couleur de fond app */
.stApp {
    background-color: #f2f4f7;
}

/* Texte dans la sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #8090b0 !important;
}

/* Boutons dans la sidebar */
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

/* Divider sidebar */
[data-testid="stSidebar"] hr {
    border-color: #1a2e48 !important;
}

/* Bouton submit form */
.stFormSubmitButton > button {
    background-color: #e8722a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
}
.stFormSubmitButton > button:hover {
    background-color: #f0914d !important;
}

/* Input text */
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

/* Messages assistant */
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
.msg-bot-label {
    font-size: 0.6rem;
    color: #5ab85c;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
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
def init_rag(_config):
    vs  = LocalVectorStore(
        persist_directory=_config["vector_store"]["persist_directory"],
        collection_name=_config["vector_store"]["collection_name"],
    )
    emb = LocalEmbeddings(_config["embeddings"]["model"])
    ret = Retriever(
        vector_store=vs, embeddings=emb,
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
    agent = RAGAgent(config=_config, retriever=ret, llm_client=llm)
    stats = vs.get_collection_stats()
    return agent, ret, stats

# ══════════════════════════════════════════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════════════════════════════════════════

for k, v in [("messages", []), ("q_count", 0), ("pending_q", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

config = load_config()

try:
    rag_agent, retriever, stats = init_rag(config)
    rag_ready = True
    doc_count = stats.get("count", 0)
except Exception:
    rag_ready = False
    doc_count = 0

examples = [
    "C'est quoi Maroclear ?",
    "Comment devenir affilié ?",
    "Qu'est-ce qu'une OPCVM ?",
    "Quel est le rôle du dépositaire central ?",
    "Quels sont les services aux affiliés ?",
    "Qu'est-ce que le règlement/livraison ?",
    "Qu'est-ce qu'un apport de titres ?",
]

model_short = config["llm"]["model"].split("/")[-1][:24]
emb_short   = config["embeddings"]["model"][:20]
now         = datetime.now().strftime("%d %b %Y · %H:%M")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NATIVE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # En-tête
    st.markdown("## 🏛️ MAROCLEAR")
    st.caption("Assistant IA · RAG v1.0")
    st.divider()

    # Statut système
    st.markdown("**STATUT SYSTÈME**")
    status = "🟢" if rag_ready else "🔴"
    st.markdown(f"{status} **Base vectorielle** — {doc_count} chunks")
    st.markdown(f"{status} **Modèle LLM** — {model_short}")
    st.markdown(f"🟢 **Embeddings** — {emb_short}")
    st.divider()

    # Actions
    st.markdown("**ACTIONS**")
    if st.button("🗑️  Effacer la conversation", key="clear_btn"):
        st.session_state.messages  = []
        st.session_state.q_count   = 0
        st.session_state.pending_q = ""
        st.rerun()
    st.divider()

    # Questions proposées
    st.markdown("**QUESTIONS PROPOSÉES**")
    for i, q in enumerate(examples):
        if st.button(q, key=f"q_{i}"):
            st.session_state.pending_q = q
            st.rerun()

    st.divider()
    st.caption(f"2026. Maroclear. Tous droits réservés.")

# ══════════════════════════════════════════════════════════════════════════════
# Traiter question depuis sidebar
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.pending_q:
    question = st.session_state.pending_q
    st.session_state.pending_q = ""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.q_count += 1

    with st.spinner("Recherche dans les documents Maroclear…"):
        try:
            response = rag_agent.process(
                AgentMessage.create_query(sender="user", content=question),
                AgentContext(),
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content,
                "sources": response.metadata.get("sources", []),
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Erreur : {e}",
                "sources": [],
            })
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:0.3rem 0;">
        <div style="background:#e8722a;width:42px;height:42px;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:1.3rem;flex-shrink:0;">
            🏛️
        </div>
        <div>
            <div style="font-size:1.3rem;font-weight:700;line-height:1.2;">
                <span style="color:#0d1b2a;">MAROC</span><span style="color:#e8722a;">CLEAR</span>
            </div>
            <div style="font-size:0.75rem;color:#6b7a94;margin-top:2px;">
                Assistant Documentaire · Dépositaire Central des Titres
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='text-align:right;color:#6b7a94;font-size:0.8rem;padding-top:0.5rem;'>{now}</div>", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ZONE CHAT
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.info("👈 Utilisez le menu latéral pour sélectionner une question, ou tapez directement ci-dessous.")

else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">'
                f'<div class="msg-user-label">Votre question</div>'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            sources  = msg.get("sources", [])
            src_text = ""
            if sources:
                names = [s.split("/")[-1].replace("_", " ") for s in sources]
                src_text = f'<div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #edf0f5;font-size:0.72rem;color:#8090b0;">📎 Sources : {" · ".join(names)}</div>'

            st.markdown(
                f'<div class="msg-bot">'
                f'<div class="msg-bot-label">● Assistant Maroclear</div>'
                f'{msg["content"].replace(chr(10), "<br>")}'
                f'{src_text}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════

st.divider()

if st.session_state.q_count > 0:
    s = "s" if st.session_state.q_count > 1 else ""
    st.caption(f"{st.session_state.q_count} question{s} posée{s} dans cette session")

with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([6, 1])
    with col_in:
        user_input = st.text_input(
            "question",
            placeholder="Posez votre question sur Maroclear…",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Envoyer →", use_container_width=True)

if submitted and user_input and user_input.strip() and rag_ready:
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.q_count += 1

    with st.spinner("Recherche dans les documents Maroclear…"):
        try:
            response = rag_agent.process(
                AgentMessage.create_query(sender="user", content=question),
                AgentContext(),
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content,
                "sources": response.metadata.get("sources", []),
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Erreur : {e}",
                "sources": [],
            })
    st.rerun()

elif submitted and not rag_ready:
    st.error("⚠️ Système RAG non initialisé. Vérifiez config.yaml.")