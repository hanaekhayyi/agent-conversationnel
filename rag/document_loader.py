"""
Chargement et traitement des documents PDF + Glossaire Excel
"""

import re
import PyPDF2
import pdfplumber
import pandas as pd
from typing import List, Dict
from pathlib import Path


class DocumentLoader:
    """Charge et extrait le texte des PDFs et du glossaire Excel"""

    def __init__(self, documents_dir: str = "data/documents"):
        self.documents_dir = Path(documents_dir)

    # =========================================================
    # PDF
    # =========================================================

    def load_pdf(self, pdf_path: Path) -> str:
        """Charge un PDF et extrait le texte"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n\n"
        except Exception as e:
            print(f"  ⚠️  pdfplumber échoué ({e}), essai avec PyPDF2...")
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n\n"
            except Exception as e2:
                print(f"  ❌ PyPDF2 aussi échoué : {e2}")
                return ""

        return self._clean_pdf_text(text)

    def _clean_pdf_text(self, text: str) -> str:
        """
        Nettoie le texte extrait d'un PDF.

        Ce PDF a un encodage particulier où les caractères accentués
        sont parfois séparés du reste du mot par un saut de ligne.
        Ex: "assur\née" → "assurée"
            "march\nés"  → "marchés"

        Stratégie : on recolle UNIQUEMENT les coupures qui précèdent
        ou suivent un caractère accentué, sans toucher aux vrais
        séparateurs de paragraphes.
        """

        # ── Étape 1 : Recoller les accents orphelins ─────────────────────────
        # Cas A : "assur\née" → le saut de ligne est AVANT la lettre accentuée
        # Le \n est entre la fin d'un mot et un début accentué
        text = re.sub(r'([a-zA-Z])\n([éèêëàâùûîïôœç])', r'\1\2', text)

        # Cas B : "é\nrer" → l'accentué est en fin de ligne, suite sur ligne suivante
        text = re.sub(r'([éèêëàâùûîïôœç])\n([a-z])', r'\1\2', text)

        # Cas C : césure classique avec tiret "produc-\ntion" → "production"
        text = re.sub(r'-\n([a-zà-ÿ])', r'\1', text)

        # ── Étape 2 : Normaliser les apostrophes et guillemets ────────────────
        text = text.replace('\u2019', "'").replace('\u2018', "'")
        text = text.replace('\u2032', "'")   # prime utilisé comme apostrophe
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('lʼ', "l'").replace('dʼ', "d'")
        text = text.replace('sʼ', "s'").replace('nʼ', "n'")
        text = text.replace('cʼ', "c'").replace('jʼ', "j'")
        text = text.replace('mʼ', "m'").replace('tʼ', "t'")

        # ── Étape 3 : Supprimer les caractères nuls et de contrôle ───────────
        # Le PDF contient des \x00 (nuls) qui apparaissent comme □ ou \u0000
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # ── Étape 4 : Supprimer les espaces multiples ─────────────────────────
        text = re.sub(r' {2,}', ' ', text)

        # ── Étape 5 : Supprimer les numéros de page isolés ───────────────────
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

        # ── Étape 6 : Réduire les sauts de ligne multiples ───────────────────
        text = re.sub(r'\n{3,}', '\n\n', text)

        # NE PAS fusionner les sauts de ligne simples → ils séparent les
        # paragraphes et sont nécessaires pour le SmartChunker

        return text.strip()

    def load_all_documents(self, exclude_keywords: List[str] = None) -> List[Dict]:
        """
        Charge tous les PDFs du répertoire.
        Exclut par défaut les fichiers contenant "glossaire" dans leur nom
        (le glossaire est chargé depuis Excel).
        """
        if exclude_keywords is None:
            exclude_keywords = ["glossaire", "glossary"]

        documents = []

        for pdf_file in self.documents_dir.glob("*.pdf"):
            filename_lower = pdf_file.name.lower()
            if any(kw in filename_lower for kw in exclude_keywords):
                print(f"  ⏭️  {pdf_file.name} ignoré (glossaire chargé depuis Excel)")
                continue

            print(f"Chargement de {pdf_file.name}...")
            text = self.load_pdf(pdf_file)

            if text:
                documents.append({
                    "filename": pdf_file.name,
                    "filepath": str(pdf_file),
                    "content":  text,
                    "metadata": {
                        "source": pdf_file.name,
                        "type":   "pdf"
                    }
                })
                print(f"  ✅ {pdf_file.name}: {len(text)} caractères extraits (après nettoyage)")

        return documents

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Découpe le texte en chunks avec overlap sur les phrases"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = ' '.join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    # =========================================================
    # GLOSSAIRE EXCEL
    # =========================================================

    def load_glossary_xlsx(self, xlsx_path: str) -> List[Dict]:
        """
        Charge le glossaire depuis un fichier Excel.
        Format attendu : colonnes "Terme" et "Définition"
        Retourne une liste de chunks prêts pour ChromaDB.
        """
        path = Path(xlsx_path)
        if not path.exists():
            print(f"❌ Fichier introuvable : {xlsx_path}")
            return []

        try:
            df = pd.read_excel(xlsx_path, engine="openpyxl")
        except Exception as e:
            print(f"❌ Erreur lecture Excel : {e}")
            return []

        df.columns = [c.strip() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}

        term_col = next(
            (col_map[k] for k in ["terme", "term", "mot", "word"] if k in col_map),
            None
        )
        def_col = next(
            (col_map[k] for k in ["définition", "definition", "def", "description"] if k in col_map),
            None
        )

        if not term_col or not def_col:
            print(f"❌ Colonnes introuvables. Disponibles : {list(df.columns)}")
            return []

        chunks = []
        skipped = 0

        for _, row in df.iterrows():
            terme      = str(row[term_col]).strip() if pd.notna(row[term_col]) else ""
            definition = str(row[def_col]).strip()  if pd.notna(row[def_col])  else ""

            if not terme or not definition or terme.lower() in ("nan", "terme", "n°"):
                skipped += 1
                continue

            chunks.append({
                "content": f"{terme}\n{definition}",
                "metadata": {
                    "term":   terme,
                    "type":   "glossaire",
                    "source": path.name,
                    "format": "xlsx"
                }
            })

        print(f"✅ Glossaire Excel : {len(chunks)} termes ({skipped} ignorées) depuis '{path.name}'")
        return chunks