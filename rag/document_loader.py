"""
Chargement et traitement des documents PDF
"""

import PyPDF2
import pdfplumber
from typing import List, Dict
from pathlib import Path

class DocumentLoader:
    """Charge et extrait le texte des PDFs"""
    
    def __init__(self, documents_dir: str = "data/documents"):
        self.documents_dir = Path(documents_dir)
    
    def load_pdf(self, pdf_path: Path) -> str:
        """Charge un PDF et extrait le texte"""
        
        text = ""
        
        # Essayer avec pdfplumber (meilleur pour les tableaux)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"Erreur pdfplumber: {e}, essai avec PyPDF2...")
            
            # Fallback sur PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n\n"
            except Exception as e2:
                print(f"Erreur PyPDF2: {e2}")
                return ""
        
        return text
    
    def load_all_documents(self) -> List[Dict]:
        """Charge tous les PDFs du répertoire"""
        
        documents = []
        
        for pdf_file in self.documents_dir.glob("*.pdf"):
            print(f"Chargement de {pdf_file.name}...")
            
            text = self.load_pdf(pdf_file)
            
            if text:
                documents.append({
                    "filename": pdf_file.name,
                    "filepath": str(pdf_file),
                    "content": text,
                    "metadata": {
                        "source": pdf_file.name,
                        "type": "pdf"
                    }
                })
                print(f"{pdf_file.name}: {len(text)} caractères extraits")
        
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Découpe le texte en chunks avec overlap
        AMÉLIORATION : Découpe sur les phrases pour préserver le sens
        """
        
        import re
        
        # Découper en phrases (approximatif)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Si ajouter la phrase dépasse chunk_size, sauvegarder le chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Garder les dernières phrases pour overlap
                overlap_text = ' '.join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Ajouter le dernier chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks