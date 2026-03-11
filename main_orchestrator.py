"""
Point d'entree principal - Systeme multi-agents Maroclear
Orchestre : ClassificationAgent + RAGAgent via OrchestratorAgent
"""

import yaml
import uuid
from pathlib import Path

# RAG Components
from rag.document_loader import DocumentLoader
from rag.smart_chunker import SmartChunker
from rag.embeddings import LocalEmbeddings
from rag.vector_store import LocalVectorStore
from rag.retriever import Retriever

# LLM
from llm.ollama_client import OllamaClient

# Agents
from agents.rag_agent import RAGAgent
from agents.classification_agent import ClassificationAgent
from agents.orchestrator_agent import OrchestratorAgent
from core.message import AgentMessage
from core.agent_context import AgentContext


# =============================================================
# CHARGEMENT CONFIG
# =============================================================

def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================
# INDEXATION DES DOCUMENTS
# =============================================================

def index_documents(config: dict):
    """Indexe les documents PDF + glossaire Excel dans ChromaDB."""

    print("\n" + "=" * 60)
    print("INDEXATION DES DOCUMENTS")
    print("=" * 60 + "\n")

    loader  = DocumentLoader(config["documents"]["sources_dir"])
    chunker = SmartChunker()

    all_chunks    = []
    all_metadatas = []
    all_ids       = []

    # Glossaire Excel
    glossary_path   = config["documents"].get(
        "glossary_xlsx", "data/documents/glossaire_maroclear.xlsx"
    )
    glossary_chunks = loader.load_glossary_xlsx(glossary_path)

    for idx, chunk in enumerate(glossary_chunks):
        all_chunks.append(chunk["content"])
        all_metadatas.append(chunk["metadata"])
        all_ids.append(f"glossaire_{idx}")

    print(f"  -> {len(glossary_chunks)} termes charges depuis le glossaire Excel\n")

    # Documents PDF
    documents = loader.load_all_documents()
    print(f"{len(documents)} document(s) PDF charge(s)\n")

    for doc in documents:
        print(f"  Chunking : {doc['filename']}...")
        raw_chunks = chunker.chunk(
            text       = doc["content"],
            source     = doc["filename"],
            chunk_size = config["rag"].get("chunk_size", 600),
            overlap    = config["rag"].get("chunk_overlap", 80),
        )
        for chunk_idx, chunk in enumerate(raw_chunks):
            all_chunks.append(chunk["content"])
            all_metadatas.append({
                **doc["metadata"],
                **chunk["metadata"],
                "chunk_index": chunk_idx,
                "doc_name":    doc["filename"],
            })
            all_ids.append(f"{uuid.uuid4()}_{doc['filename']}_chunk_{chunk_idx}")
        print(f"     -> {len(raw_chunks)} chunks")

    print(f"\nTotal : {len(all_chunks)} chunks")

    # Embeddings
    embeddings_model = LocalEmbeddings(config["embeddings"]["model"])
    embeddings = embeddings_model.embed_documents(
        all_chunks,
        batch_size=config["embeddings"]["batch_size"]
    )

    # Stockage ChromaDB
    vector_store = LocalVectorStore(
        persist_directory=config["vector_store"]["persist_directory"],
        collection_name=config["vector_store"]["collection_name"],
    )
    print("Nettoyage de la collection existante...")
    vector_store.clear_collection()
    vector_store.add_documents(
        texts=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print(f"\nVerification : {vector_store.get_collection_stats()}")
    print("\n" + "=" * 60)
    print("INDEXATION TERMINEE")
    print("=" * 60 + "\n")

    return vector_store, embeddings_model


# =============================================================
# CONSTRUCTION DE L'ORCHESTRATEUR
# =============================================================

def build_orchestrator(config: dict, vector_store=None, embeddings_model=None):
    """
    Instancie les trois agents et retourne l'orchestrateur pret a l'emploi.
    """

    # Charger le vector store si non fourni
    if not vector_store:
        vector_store = LocalVectorStore(
            persist_directory=config["vector_store"]["persist_directory"],
            collection_name=config["vector_store"]["collection_name"],
        )

    if not embeddings_model:
        embeddings_model = LocalEmbeddings(config["embeddings"]["model"])

    # Retriever pour le RAGAgent
    retriever = Retriever(
        vector_store=vector_store,
        embeddings=embeddings_model,
        top_k=config["rag"].get("top_k", 8),
        similarity_threshold=config["rag"].get("similarity_threshold", 0.4),
    )

    # Client LLM partage entre les deux agents
    llm_config = config["llm"]
    llm_client = OllamaClient(
        model=llm_config["model"],
        temperature=llm_config.get("temperature", 0.1),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url", "https://openrouter.ai/api/v1"),
        max_tokens=llm_config.get("max_tokens", 1024),
    )

    # Instanciation des agents
    rag_agent = RAGAgent(
        config=config,
        retriever=retriever,
        llm_client=llm_client,
    )

    classification_agent = ClassificationAgent(
        config=config,
        llm_client=llm_client,
    )

    # Orchestrateur
    orchestrator = OrchestratorAgent(
        config=config,
        classification_agent=classification_agent,
        rag_agent=rag_agent,
    )

    return orchestrator


# =============================================================
# BOUCLE DE CONVERSATION INTERACTIVE
# =============================================================

def run_interactive(orchestrator: OrchestratorAgent):
    """
    Lance une session de conversation interactive en terminal.
    Chaque message passe par l'orchestrateur qui decide quel agent appeler.
    """

    print("\n" + "=" * 60)
    print("  MAROCLEAR - Systeme Multi-Agents")
    print("  Agents disponibles : Classification + RAG")
    print("=" * 60)
    print("  Commandes : 'quitter' | 'reset' | 'status' | 'capacites'")
    print("=" * 60 + "\n")

    print("Bonjour ! Je suis l'assistant Maroclear.")
    print("Je peux vous aider avec :")
    print("  - Un incident technique (blocage, dysfonctionnement)")
    print("  - Une demande de service (document, acces, PIN...)")
    print("  - Une reclamation")
    print("  - Une question sur les services ou procedures Maroclear\n")

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminee.")
            break

        if not user_input:
            continue

        # Commandes speciales
        if user_input.lower() in ("quitter", "exit", "quit"):
            print("\nMerci d'avoir utilise l'assistant Maroclear. Au revoir !")
            break

        if user_input.lower() == "reset":
            orchestrator.reset_context()
            print("\nConversation reinitalisee.\n")
            continue

        if user_input.lower() == "status":
            history = orchestrator.context.get_history()
            print(f"\nStatut de la session :")
            print(f"  Messages echanges : {len(history)}")
            print(f"  Dernier agent     : {orchestrator.context.current_agent or 'aucun'}\n")
            continue

        if user_input.lower() == "capacites":
            caps = orchestrator.get_capabilities()
            print(f"\nCapacites de l'orchestrateur :")
            for route, desc in caps["routes"].items():
                print(f"  [{route}] {desc}")
            print()
            continue

        # Traitement du message
        print()
        result = orchestrator.chat(user_input)

        print(f"Assistant [{result['agent_used']}] :\n")
        print(result["response"])

        # Afficher le resume de classification si disponible
        if result["classification"]:
            print("\n" + "-" * 40)
            print(f"  Agent utilise  : {result['agent_used']}")
            print(f"  Classification : {result['classification']}")
            if result["priorite"]:
                print(f"  Priorite       : {result['priorite']}")
            if result["statut"]:
                print(f"  Statut         : {result['statut']}")
            if result["needs_glpi"]:
                print(f"  Ticket GLPI    : A creer")
            if result["rag_sources"]:
                print(f"  Sources RAG    : {result['rag_sources']}")
            print("-" * 40)

        print()


# =============================================================
# POINT D'ENTREE
# =============================================================

def main():
    config = load_config()

    print("\n" + "=" * 60)
    print("  MAROCLEAR - SYSTEME MULTI-AGENTS")
    print("=" * 60)
    print("\nQue voulez-vous faire ?")
    print("  1. Indexer les documents")
    print("  2. Lancer la conversation (documents deja indexes)")
    print("  3. Indexer ET lancer la conversation")

    choice = input("\nVotre choix (1/2/3) : ").strip()

    if choice == "1":
        index_documents(config)

    elif choice == "2":
        orchestrator = build_orchestrator(config)
        run_interactive(orchestrator)

    elif choice == "3":
        vector_store, embeddings_model = index_documents(config)
        orchestrator = build_orchestrator(config, vector_store, embeddings_model)
        run_interactive(orchestrator)

    else:
        print("Choix invalide.")


if __name__ == "__main__":
    main()
