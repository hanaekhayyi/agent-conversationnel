"""
Script de diagnostic du chunking - sauvegarde JSON pour inspection visuelle.
Lancer SÉPARÉMENT de main.py pour déboguer avant d'indexer.

Inspecte :
  - Le glossaire Excel  → data/chunks_debug/glossaire_xlsx_chunks.json
  - Le PDF Maroclear    → data/chunks_debug/<nom_pdf>_chunks.json
  - Tout combiné        → data/chunks_debug/ALL_chunks_combined.json

Usage:
    python debug_chunks.py
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────
#  SMART CHUNKER (VERSION FINALE)
#  - Correction faux positif GLOSSARY_INLINE (seuil 0.50)
#  - type='narrative' sur TOUS les chunks PDF (jamais 'glossaire')
#  - 'glossaire' réservé exclusivement au glossaire Excel
# ─────────────────────────────────────────────────────────────

class DocumentFormat(Enum):
    GLOSSARY_NEWLINE = "glossary_newline"
    GLOSSARY_COLON   = "glossary_colon"
    GLOSSARY_INLINE  = "glossary_inline"
    NARRATIVE        = "narrative"
    SECTIONED        = "sectioned"


class SmartChunker:

    def detect_format(self, text: str) -> DocumentFormat:
        sample = text[:5000]
        lines  = [l.strip() for l in sample.split("\n") if l.strip()]
        total  = len(lines)
        if total == 0:
            return DocumentFormat.NARRATIVE

        short_followed_by_long = 0
        colon_pattern          = 0
        section_markers        = 0
        inline_glued           = 0

        for i, line in enumerate(lines):
            if len(line) < 80 and i + 1 < total and len(lines[i + 1]) > 80:
                short_followed_by_long += 1
            if re.match(r'^[^:]{3,60}:\s+[A-ZÀ-Ÿ]', line):
                colon_pattern += 1
            if re.match(r'^(›|#{1,3}\s|[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ\s]{6,}$)', line):
                section_markers += 1
            if re.search(r'[a-zà-ÿ]{3,}[A-ZÀ-Ÿ][a-zà-ÿ]', line):
                inline_glued += 1

        scores = {
            DocumentFormat.GLOSSARY_NEWLINE : short_followed_by_long / total,
            DocumentFormat.GLOSSARY_COLON   : colon_pattern          / total,
            DocumentFormat.GLOSSARY_INLINE  : inline_glued           / total,
            DocumentFormat.SECTIONED        : section_markers        / total,
        }

        print(f"  📊 Scores de détection :")
        for fmt, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"     {fmt.value:<22} {score:.3f}")

        best  = max(scores, key=scores.get)
        score = scores[best]

        # Correction faux positif GLOSSARY_INLINE
        # INLINE score faussement ~1.0 sur tout texte français.
        # Seuil NEWLINE à 0.25 : un vrai glossaire a beaucoup de termes courts
        # suivis de longues définitions. Un narratif peut scorer 0.10-0.15
        # accidentellement (sous-titres courts + paragraphes longs).
        if best == DocumentFormat.GLOSSARY_INLINE:
            if scores[DocumentFormat.GLOSSARY_NEWLINE] >= 0.25:
                best = DocumentFormat.GLOSSARY_NEWLINE
                print("  ⚠️  INLINE → NEWLINE (signal newline fort, vrai glossaire)")
            elif scores[DocumentFormat.SECTIONED] >= 0.05:
                best = DocumentFormat.SECTIONED
                print("  ⚠️  INLINE → SECTIONED (marqueurs de sections détectés)")
            elif score < 0.50:
                best = DocumentFormat.NARRATIVE
                print("  ⚠️  INLINE → NARRATIVE (score ambigu, défaut narratif)")

        # Vérification finale uniquement si best est dans scores
        if best in scores and scores[best] < 0.04:
            best = DocumentFormat.NARRATIVE

        print(f"  ✅ Format retenu : {best.value}\n")
        return best

    def chunk(
        self,
        text: str,
        source: str = "unknown",
        force_format: Optional[DocumentFormat] = None,
        chunk_size: int = 600,
        overlap: int = 80,
    ) -> List[Dict]:
        fmt = force_format or self.detect_format(text)

        dispatch = {
            DocumentFormat.GLOSSARY_NEWLINE : self._chunk_glossary_newline,
            DocumentFormat.GLOSSARY_COLON   : self._chunk_glossary_colon,
            DocumentFormat.GLOSSARY_INLINE  : self._chunk_glossary_inline,
            DocumentFormat.SECTIONED        : lambda t: self._chunk_sectioned(t, chunk_size),
            DocumentFormat.NARRATIVE        : lambda t: self._chunk_narrative(t, chunk_size, overlap),
        }

        chunks = dispatch[fmt](text)

        for chunk in chunks:
            chunk["metadata"]["source"] = source
            chunk["metadata"]["format"] = fmt.value
            # ✅ CORRECTION CLEF : type='glossaire' interdit sur chunks PDF
            if chunk["metadata"].get("type") == "glossaire":
                chunk["metadata"]["type"] = "narrative"

        return chunks

    # ── Stratégies ───────────────────────────────────────────

    def _chunk_glossary_newline(self, text: str) -> List[Dict]:
        lines = [l.strip() for l in text.split("\n")]
        chunks, current_term, current_def = [], None, []
        for line in lines:
            if not line:
                continue
            if self._is_term_line(line):
                if current_term and current_def:
                    chunks.append(self._make_chunk(current_term, current_def))
                current_term = line
                current_def  = []
            else:
                current_def.append(line)
        if current_term and current_def:
            chunks.append(self._make_chunk(current_term, current_def))
        return chunks

    def _chunk_glossary_colon(self, text: str) -> List[Dict]:
        chunks = []
        for line in text.split("\n"):
            line  = line.strip()
            match = re.match(r'^([^:]{3,60}):\s+(.+)', line)
            if match:
                chunks.append(self._make_chunk(match.group(1), [match.group(2)]))
        return chunks

    def _chunk_glossary_inline(self, text: str) -> List[Dict]:
        text = re.sub(
            r'([A-ZÀ-Ÿa-zà-ÿ/\-()\''']{3,})([A-ZÀ-Ÿ][a-zà-ÿ])',
            lambda m: m.group(1) + "\n" + m.group(2),
            text
        )
        return self._chunk_glossary_newline(text)

    def _chunk_sectioned(self, text: str, chunk_size: int = 800) -> List[Dict]:
        pattern = re.compile(
            r'(?m)(?=^(?:›\s+|\#{1,3}\s+|[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ\s]{5,}$))'
        )
        chunks = []
        for section in pattern.split(text):
            section = section.strip()
            if not section:
                continue
            if len(section) > chunk_size:
                for sub in self._narrative_raw(section, chunk_size):
                    chunks.append({"content": sub, "metadata": {"type": "narrative"}})
            else:
                chunks.append({"content": section, "metadata": {"type": "section"}})
        return chunks

    def _chunk_narrative(self, text: str, chunk_size: int = 600,
                         overlap: int = 80) -> List[Dict]:
        return [
            {"content": c, "metadata": {"type": "narrative"}}
            for c in self._narrative_raw(text, chunk_size, overlap)
        ]

    def _narrative_raw(self, text: str, chunk_size: int = 600,
                       overlap: int = 80) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size and current:
                chunks.append(current.strip())
                # Overlap en MOTS pour éviter les coupures en milieu de mot
                last_words    = current.split()
                overlap_words = last_words[-overlap:] if len(last_words) > overlap else last_words
                current = " ".join(overlap_words) + " " + sentence
            else:
                current += " " + sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _is_term_line(self, line: str) -> bool:
        if not line or len(line) > 100:
            return False
        definition_starters = (
            "le ", "la ", "les ", "un ", "une ", "des ",
            "il ", "elle ", "on ", "ce ", "cette ", "ces ",
            "toute ", "tout ", "dans ", "lorsque ", "se dit",
            "correspond", "désigne ", "sont ", "permet ",
            "c'est ", "cʼest ", "en ", "par ", "pour ",
            "selon ", "dont ", "qui ", "que ", "lʼ", "l'",
        )
        lower = line.lower()
        if any(lower.startswith(s) for s in definition_starters):
            return False
        if line.endswith(".") and len(line) > 30:
            return False
        if "," in line and len(line) > 60:
            return False
        return True

    def _make_chunk(self, term: str, definition: List[str]) -> Dict:
        # type='narrative' ici car ce chunker ne traite QUE des PDFs
        # Le type 'glossaire' est réservé au glossaire Excel (load_glossary_xlsx)
        return {
            "content" : f"{term}\n{' '.join(definition)}",
            "metadata": {"term": term, "type": "narrative"},
        }


# ─────────────────────────────────────────────────────────────
#  NETTOYAGE TEXTE PDF
#  - Recollent les accents orphelins ("assur\née" → "assurée")
#  - Recollent les césures ("-\ntion" → "tion")
#  - Normalise les apostrophes typographiques
#  - NE fusionne PAS les sauts de ligne simples (structure préservée)
# ─────────────────────────────────────────────────────────────

def clean_pdf_text(text: str) -> str:
    # "assur\née" → "assurée"
    text = re.sub(r'([a-zA-Z])\n([éèêëàâùûîïôœç])', r'\1\2', text)
    # "é\nrer"    → "érer"
    text = re.sub(r'([éèêëàâùûîïôœç])\n([a-z])', r'\1\2', text)
    # "produc-\ntion" → "production"
    text = re.sub(r'-\n([a-zà-ÿ])', r'\1', text)

    # Apostrophes typographiques → apostrophe standard
    for old, new in [
        ('\u2019', "'"), ('\u2018', "'"), ('\u2032', "'"),
        ('lʼ', "l'"), ('dʼ', "d'"), ('sʼ', "s'"), ('nʼ', "n'"),
        ('cʼ', "c'"), ('jʼ', "j'"), ('mʼ', "m'"), ('tʼ', "t'"),
    ]:
        text = text.replace(old, new)

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # nuls et contrôles
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_pdf_text(pdf_path: str) -> str:
    """Extrait et nettoie le texte d'un PDF."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n\n"
        return clean_pdf_text(text)
    except Exception as e:
        print(f"  pdfplumber échoué ({e}), essai PyPDF2...")
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n\n"
            return clean_pdf_text(text)
        except Exception as e2:
            print(f"  PyPDF2 aussi échoué : {e2}")
            return ""


# ─────────────────────────────────────────────────────────────
#  CHARGEMENT GLOSSAIRE EXCEL
#  - type='glossaire' UNIQUEMENT ici
#  - source = nom du fichier .xlsx
# ─────────────────────────────────────────────────────────────

def load_glossary_xlsx(xlsx_path: str) -> List[Dict]:
    """Charge le glossaire Excel → chunks avec type='glossaire'."""
    path = Path(xlsx_path)
    if not path.exists():
        print(f"  ❌ Fichier introuvable : {xlsx_path}")
        return []
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception as e:
        print(f"  ❌ Erreur lecture Excel : {e}")
        return []

    df.columns = [c.strip() for c in df.columns]
    col_map  = {c.lower(): c for c in df.columns}
    term_col = next((col_map[k] for k in ["terme","term","mot","word"] if k in col_map), None)
    def_col  = next((col_map[k] for k in ["définition","definition","def","description"] if k in col_map), None)

    if not term_col or not def_col:
        print(f"  ❌ Colonnes introuvables. Disponibles : {list(df.columns)}")
        return []

    chunks, skipped = [], 0
    for _, row in df.iterrows():
        terme      = str(row[term_col]).strip() if pd.notna(row[term_col]) else ""
        definition = str(row[def_col]).strip()  if pd.notna(row[def_col])  else ""
        if not terme or not definition or terme.lower() in ("nan", "terme", "n°"):
            skipped += 1
            continue
        chunks.append({
            "content" : f"{terme}\n{definition}",
            "metadata": {
                "term"  : terme,
                "type"  : "glossaire",   # ✅ seul endroit où type='glossaire'
                "source": path.name,
                "format": "xlsx",
            }
        })

    print(f"  ✅ {len(chunks)} termes ({skipped} ignorés) depuis '{path.name}'")
    return chunks


# ─────────────────────────────────────────────────────────────
#  SAUVEGARDE JSON + STATISTIQUES
# ─────────────────────────────────────────────────────────────

def save_chunks_to_json(chunks: List[Dict], output_path: str):
    """Sauvegarde les chunks dans un JSON lisible avec stats intégrées."""
    lengths = [len(c["content"]) for c in chunks] if chunks else [0]
    by_type = {}
    for c in chunks:
        t = c["metadata"].get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    output = {
        "total_chunks"  : len(chunks),
        "stats": {
            "by_type"       : by_type,
            "length_min"    : min(lengths),
            "length_max"    : max(lengths),
            "length_avg"    : sum(lengths) // len(lengths) if lengths else 0,
        },
        "chunks": [
            {
                "index"   : i,
                "preview" : c["content"][:120].replace("\n", " ↵ "),
                "length"  : len(c["content"]),
                "metadata": c["metadata"],
                "content" : c["content"],
            }
            for i, c in enumerate(chunks)
        ]
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  💾 {output_path}")


def print_chunk_stats(chunks: List[Dict], label: str):
    if not chunks:
        print(f"  ⚠️  Aucun chunk pour {label}")
        return

    lengths = [len(c["content"]) for c in chunks]
    by_type = {}
    for c in chunks:
        t = c["metadata"].get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\n  📈 Statistiques — {label}")
    print(f"     Chunks          : {len(chunks)}")
    print(f"     Longueur min    : {min(lengths)} chars")
    print(f"     Longueur max    : {max(lengths)} chars")
    print(f"     Longueur moy.   : {sum(lengths)//len(lengths)} chars")
    print(f"     Types           : {by_type}")

    print(f"\n  🔍 3 premiers chunks :")
    for c in chunks[:3]:
        preview = c["content"][:150].replace("\n", " ↵ ")
        print(f"     [{c['metadata'].get('type','?')}] {preview}")

    print(f"\n  🔍 3 derniers chunks :")
    for c in chunks[-3:]:
        preview = c["content"][:150].replace("\n", " ↵ ")
        print(f"     [{c['metadata'].get('type','?')}] {preview}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    DOCS_DIR         = Path("data/documents")
    OUTPUT_DIR       = Path("data/chunks_debug")
    GLOSSARY_XLSX    = "data/documents/glossaire_maroclear.xlsx"
    CHUNK_SIZE       = 600
    OVERLAP          = 80
    # Fichiers PDF à ignorer (glossaire chargé depuis Excel)
    EXCLUDE_KEYWORDS = ["glossaire", "glossary"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunker = SmartChunker()
    all_chunks_combined = []

    print("\n" + "="*60)
    print("  DIAGNOSTIC DU CHUNKING")
    print("="*60)

    # ── 1. Glossaire Excel ────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📊 GLOSSAIRE EXCEL : {GLOSSARY_XLSX}")
    print(f"{'─'*60}")

    glossary_chunks = load_glossary_xlsx(GLOSSARY_XLSX)
    print_chunk_stats(glossary_chunks, "Glossaire Excel")
    save_chunks_to_json(glossary_chunks, str(OUTPUT_DIR / "glossaire_xlsx_chunks.json"))
    all_chunks_combined.extend(glossary_chunks)

    # ── 2. PDFs (hors fichiers glossaire) ─────────────────────
    pdf_files = [
        f for f in DOCS_DIR.glob("*.pdf")
        if not any(kw in f.name.lower() for kw in EXCLUDE_KEYWORDS)
    ]

    if not pdf_files:
        print(f"\n  ⚠️  Aucun PDF trouvé dans {DOCS_DIR} (hors glossaire)")
    else:
        for pdf_path in pdf_files:
            print(f"\n{'─'*60}")
            print(f"  📄 PDF : {pdf_path.name}")
            print(f"{'─'*60}")

            text = load_pdf_text(str(pdf_path))
            if not text:
                print("  ❌ Texte vide, ignoré")
                continue

            print(f"  📝 {len(text)} chars extraits (après nettoyage)")
            print(f"  🔎 Détection automatique du format...")

            chunks = chunker.chunk(
                text=text,
                source=pdf_path.name,
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
            )

            print_chunk_stats(chunks, pdf_path.name)
            json_path = OUTPUT_DIR / f"{pdf_path.stem[:40]}_chunks.json"
            save_chunks_to_json(chunks, str(json_path))
            all_chunks_combined.extend(chunks)

    # ── 3. JSON combiné ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  📦 RÉSUMÉ GLOBAL")
    print(f"{'─'*60}")

    by_type, by_source = {}, {}
    for c in all_chunks_combined:
        t = c["metadata"].get("type","?")
        s = c["metadata"].get("source","?")
        by_type[t]   = by_type.get(t,0) + 1
        by_source[s] = by_source.get(s,0) + 1

    print(f"  Total chunks : {len(all_chunks_combined)}")
    print(f"  Par type     : {by_type}")
    print(f"  Par source   : {by_source}")

    save_chunks_to_json(all_chunks_combined, str(OUTPUT_DIR / "ALL_chunks_combined.json"))

    print(f"\n✅ Fichiers disponibles dans : {OUTPUT_DIR}/")
    print("   → glossaire_xlsx_chunks.json  : définitions du glossaire Excel")
    print("   → <nom_pdf>_chunks.json       : chunks du PDF Maroclear")
    print("   → ALL_chunks_combined.json    : tout ce qui ira dans ChromaDB")
    print("\n   📌 Critères de qualité à vérifier dans le JSON :")
    print("      ✅ Chunks glossaire : type='glossaire', source='*.xlsx'")
    print("      ✅ Chunks PDF       : type='narrative', source='*.pdf'")
    print("      ✅ Contenu lisible  : pas d'accents cassés, pas de mots collés")
    print("      ✅ Longueur         : 200–800 chars (trop court = perte d'info)\n")


if __name__ == "__main__":
    main()