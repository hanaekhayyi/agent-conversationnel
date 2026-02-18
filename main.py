"""
Script principal pour indexer les documents et tester le RAG Agent
"""

import yaml 
import uuid
from pathlib import Path

# RAG Components
from rag.document_loader import DocumentLoader
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
    """Indexe les documents PDF dans ChromaDB"""
    
    print("\n" + "="*60)
    print("INDEXATION DES DOCUMENTS")
    print("="*60 + "\n")
    
    # 1. Charger les documents
    loader = DocumentLoader(config['documents']['sources_dir'])
    documents = loader.load_all_documents()
    
    if not documents:
        print("Aucun document trouvé !")
        return None
    
    print(f"\n{len(documents)} documents chargés\n")
    
    # 2. Chunking
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    chunk_size = config['rag']['chunk_size']
    chunk_overlap = config['rag']['chunk_overlap']
    
    for doc_idx, doc in enumerate(documents):
        chunks = loader.chunk_text(
            doc['content'],
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                **doc['metadata'],
                "chunk_index": chunk_idx,
                "doc_name": doc['filename']
            })
            # Créer des IDs uniques avec UUID pour éviter les doublons
            unique_id = f"{uuid.uuid4()}_{doc['filename']}_chunk_{chunk_idx}"
            all_ids.append(unique_id)
        
        print(f"{doc['filename']}: {len(chunks)} chunks")
    
    print(f"\nTotal: {len(all_chunks)} chunks\n")
    
    # 3. Générer embeddings
    embeddings_model = LocalEmbeddings(config['embeddings']['model'])
    embeddings = embeddings_model.embed_documents(
        all_chunks,
        batch_size=config['embeddings']['batch_size']
    )
    
    # 4. Stocker dans ChromaDB
    vector_store = LocalVectorStore(
        persist_directory=config['vector_store']['persist_directory'],
        collection_name=config['vector_store']['collection_name']
    )
    
    # Vider la collection avant d'ajouter de nouveaux documents
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

    # Créer le retriever
    retriever = Retriever(
        vector_store=vector_store,
        embeddings=embeddings_model,
        top_k=config['rag']['top_k'],
        similarity_threshold=config['rag']['similarity_threshold']  # 🔧 MODIFIÉ
    )
    
    # Si pas de vector store fourni, en créer un
    if not vector_store:
        vector_store = LocalVectorStore(
            persist_directory=config['vector_store']['persist_directory'],
            collection_name=config['vector_store']['collection_name']
        )
    
    if not embeddings_model:
        embeddings_model = LocalEmbeddings(config['embeddings']['model'])
    
    # Créer le retriever
    retriever = Retriever(
        vector_store=vector_store,
        embeddings=embeddings_model,
        top_k=config['rag']['top_k'],
        similarity_threshold=config['rag']['similarity_threshold']
    )
    
    # Créer le client LLM
    llm_client = OllamaClient(
        model=config['llm']['model'],
        temperature=config['llm']['temperature']
    )
    
    # Créer le RAG Agent
    rag_agent = RAGAgent(
        config=config,
        retriever=retriever,
        llm_client=llm_client
    )
    
    # Questions de test
    test_questions = [
        "C'est quoi Maroclear ?",
        "Quels sont les services proposés aux affiliés ?",
        "Comment devenir affilié chez Maroclear ?",
        "Qu'est-ce qu'une OPCVM ?",  # Définition du glossaire
        "Quel est le rôle du dépositaire central ?",
    ]
    
    context = AgentContext()
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        # Créer message
        message = AgentMessage.create_query(
            sender="user",
            content=question
        )
        
        # Vérifier si l'agent peut traiter
        if rag_agent.can_handle(message, context):
            # Traiter
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
    
    # Choix utilisateur
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