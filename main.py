"""
Script principal pour indexer les documents et tester le RAG Agent
"""

import yaml 
import uuid
from pathlib import Path

# RAG Components
from rag.document_loader import DocumentLoader
from rag.smart_chunker import SmartChunker, DocumentFormat
from rag.embeddings import LocalEmbeddings
from rag.vector_store import LocalVectorStore
from rag.retriever import Retriever

# LLM
from llm.ollama_client import OllamaClient

# Agent
from agents.rag_agent import RAGAgent
from core.message import AgentMessage
from core.agent_context import AgentContext

def load_config():
    """Charge la configuration"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def index_documents(config):
    """Indexe les documents PDF + glossaire Excel dans ChromaDB"""
    
    print("\n" + "="*60)
    print("INDEXATION DES DOCUMENTS")
    print("="*60 + "\n")
    
    loader  = DocumentLoader(config['documents']['sources_dir'])
    chunker = SmartChunker()
    
    all_chunks    = []
    all_metadatas = []
    all_ids       = []

    # ─────────────────────────────────────────────────────────
    # 1. Glossaire Excel  (1 ligne = 1 chunk, aucune heuristique)
    # ─────────────────────────────────────────────────────────
    glossary_path = config['documents'].get(
        'glossary_xlsx',
        'data/documents/glossaire_maroclear.xlsx'
    )

    glossary_chunks = loader.load_glossary_xlsx(glossary_path)

    for idx, chunk in enumerate(glossary_chunks):
        all_chunks.append(chunk['content'])
        all_metadatas.append(chunk['metadata'])
        all_ids.append(f"glossaire_{idx}")

    print(f"  → {len(glossary_chunks)} termes chargés depuis le glossaire Excel\n")

    # ─────────────────────────────────────────────────────────
    # 2. Documents PDF  (SmartChunker avec détection automatique)
    # ─────────────────────────────────────────────────────────
    documents = loader.load_all_documents()

    if not documents and not glossary_chunks:
        print("Aucun document trouvé !")
        return None

    print(f"{len(documents)} document(s) PDF chargé(s)\n")

    for doc in documents:
        print(f"  📄 Chunking : {doc['filename']}...")

        raw_chunks = chunker.chunk(
            text       = doc['content'],
            source     = doc['filename'],
            chunk_size = config['rag'].get('chunk_size', 600),
            overlap    = config['rag'].get('chunk_overlap', 80),
        )

        for chunk_idx, chunk in enumerate(raw_chunks):
            all_chunks.append(chunk['content'])
            all_metadatas.append({
                **doc['metadata'],
                **chunk['metadata'],
                "chunk_index": chunk_idx,
                "doc_name":    doc['filename'],
            })
            all_ids.append(f"{uuid.uuid4()}_{doc['filename']}_chunk_{chunk_idx}")

        print(f"     → {len(raw_chunks)} chunks")

    print(f"\nTotal : {len(all_chunks)} chunks")
    print(f"  dont {len(glossary_chunks)} termes glossaire Excel")
    print(f"  dont {len(all_chunks) - len(glossary_chunks)} chunks PDF\n")

    # ─────────────────────────────────────────────────────────
    # 3. Embeddings
    # ─────────────────────────────────────────────────────────
    embeddings_model = LocalEmbeddings(config['embeddings']['model'])
    embeddings = embeddings_model.embed_documents(
        all_chunks,
        batch_size=config['embeddings']['batch_size']
    )

    # ─────────────────────────────────────────────────────────
    # 4. Stockage ChromaDB
    # ─────────────────────────────────────────────────────────
    vector_store = LocalVectorStore(
        persist_directory=config['vector_store']['persist_directory'],
        collection_name=config['vector_store']['collection_name']
    )

    print("Nettoyage de la collection existante...")
    vector_store.clear_collection()

    vector_store.add_documents(
        texts=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print(f"\nVérification: {vector_store.get_collection_stats()}")

    print("\n" + "="*60)
    print("INDEXATION TERMINÉE")
    print("="*60 + "\n")

    return vector_store, embeddings_model


def test_rag_agent(config, vector_store=None, embeddings_model=None):
    """Test du RAG Agent"""
    
    print("\n" + "="*60)
    print("TEST DU RAG AGENT")
    print("="*60 + "\n")

    if not vector_store:
        vector_store = LocalVectorStore(
            persist_directory=config['vector_store']['persist_directory'],
            collection_name=config['vector_store']['collection_name']
        )

    if not embeddings_model:
        embeddings_model = LocalEmbeddings(config['embeddings']['model'])

    # ✅ MODIFIÉ : .get() avec valeurs par défaut alignées sur les nouveaux fichiers
    # top_k          : 8   (était 5)
    # threshold      : 0.4 (était 0.5)
    # temperature    : 0.1 (était 0.2)
    retriever = Retriever(
        vector_store=vector_store,
        embeddings=embeddings_model,
        top_k=config['rag'].get('top_k', 8),
        similarity_threshold=config['rag'].get('similarity_threshold', 0.4),
    )
    """
    llm_client = OllamaClient(
        model=config['llm']['model'],
        temperature=config['llm'].get('temperature', 0.1),
    )
    """
    llm_client = OllamaClient(
    model=config['llm']['model'],
    temperature=config['llm'].get('temperature', 0.1),
    api_key=config['llm'].get('api_key'),
    base_url=config['llm'].get('base_url', 'https://openrouter.ai/api/v1'),
    max_tokens=config['llm'].get('max_tokens', 1024),
    )

    rag_agent = RAGAgent(
        config=config,
        retriever=retriever,
        llm_client=llm_client
    )

    test_questions = [
        #"C'est quoi Maroclear ?",
        #"Quels sont les services proposés aux affiliés ?",
        #"Comment devenir affilié chez Maroclear ?",
        #"Qu'est-ce qu'un Apport de titres ?",
        #"Quel est le rôle du dépositaire central ?",
        "Qu'est-ce qu'un OPCVM ?",
    ]

    context = AgentContext()

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)

        message = AgentMessage.create_query(
            sender="user",
            content=question
        )

        if rag_agent.can_handle(message, context):
            response = rag_agent.process(message, context)
            print(f"Réponse:\n{response.content}")
            print(f"\nSources: {response.metadata.get('sources', [])}")
        else:
            print("RAG Agent ne peut pas traiter cette question")

        print("\n" + "="*60)


def main():
    """Point d'entrée principal"""
    
    config = load_config()
    
    print("\n" + "="*60)
    print("MAROCLEAR RAG AGENT - SYSTÈME MULTI-AGENT")
    print("="*60)
    
    print("\nQue voulez-vous faire ?")
    print("1. Indexer les documents")
    print("2. Tester le RAG Agent")
    print("3. Indexer ET tester")
    
    choice = input("\nVotre choix (1/2/3): ").strip()
    
    if choice == "1":
        index_documents(config)
    elif choice == "2":
        test_rag_agent(config)
    elif choice == "3":
        vector_store, embeddings_model = index_documents(config)
        test_rag_agent(config, vector_store, embeddings_model)
    else:
        print("Choix invalide")

if __name__ == "__main__":
    main()