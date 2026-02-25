"""
SmartChunker adaptatif - version avec support glossaire Excel.

Le glossaire Excel est traité en amont par DocumentLoader.load_glossary_xlsx().
SmartChunker est appelé UNIQUEMENT sur les documents PDF Maroclear.

IMPORTANT : Les chunks produits par SmartChunker reçoivent type='narrative'
ou type='section', jamais type='glossaire' (réservé au glossaire Excel).
"""

from enum import Enum
from typing import List, Dict, Optional
import re


class DocumentFormat(Enum):
    GLOSSARY_NEWLINE = "glossary_newline"
    GLOSSARY_COLON   = "glossary_colon"
    GLOSSARY_INLINE  = "glossary_inline"
    NARRATIVE        = "narrative"
    SECTIONED        = "sectioned"


class SmartChunker:

    # =========================================================
    # DÉTECTION AUTOMATIQUE DU FORMAT
    # =========================================================

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
            DocumentFormat.GLOSSARY_NEWLINE: short_followed_by_long / total,
            DocumentFormat.GLOSSARY_COLON:   colon_pattern          / total,
            DocumentFormat.GLOSSARY_INLINE:  inline_glued           / total,
            DocumentFormat.SECTIONED:        section_markers        / total,
        }

        print(f"  📊 Scores de détection :")
        for fmt, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"     {fmt.value:<22} {score:.3f}")

        best  = max(scores, key=scores.get)
        score = scores[best]

        # ── Correction du faux positif GLOSSARY_INLINE ───────────────────────
        # INLINE score faussement ~1.0 sur TOUT texte français (pattern trop large)
        # NEWLINE seuil à 0.25 : un vrai glossaire a beaucoup de "terme court
        # suivi d'une définition longue". Un narratif peut scorer 0.10-0.15
        # accidentellement via des sous-titres courts suivis de paragraphes.
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

        if best in scores and scores[best] < 0.04:
            best = DocumentFormat.NARRATIVE

        print(f"  ✅ Format retenu : {best.value}\n")
        return best

    # =========================================================
    # POINT D'ENTRÉE UNIQUE
    # =========================================================

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
            DocumentFormat.GLOSSARY_NEWLINE: self._chunk_glossary_newline,
            DocumentFormat.GLOSSARY_COLON:   self._chunk_glossary_colon,
            DocumentFormat.GLOSSARY_INLINE:  self._chunk_glossary_inline,
            DocumentFormat.SECTIONED:        lambda t: self._chunk_sectioned(t, chunk_size),
            DocumentFormat.NARRATIVE:        lambda t: self._chunk_narrative(t, chunk_size, overlap),
        }

        chunks = dispatch[fmt](text)

        for chunk in chunks:
            chunk["metadata"]["source"] = source
            chunk["metadata"]["format"] = fmt.value
            # ✅ CORRECTION CLEF : écraser le type pour ne jamais avoir
            # 'glossaire' sur un chunk PDF — réservé au glossaire Excel
            if chunk["metadata"].get("type") == "glossaire":
                chunk["metadata"]["type"] = "narrative"

        return chunks

    # =========================================================
    # STRATÉGIES
    # =========================================================

    def _chunk_glossary_newline(self, text: str) -> List[Dict]:
        lines = [l.strip() for l in text.split("\n")]
        chunks, current_term, current_def = [], None, []
        for line in lines:
            if not line:
                continue
            if self._is_term_line(line):
                if current_term and current_def:
                    chunks.append(self._make_glossary_chunk(current_term, current_def))
                current_term = line
                current_def  = []
            else:
                current_def.append(line)
        if current_term and current_def:
            chunks.append(self._make_glossary_chunk(current_term, current_def))
        return chunks

    def _chunk_glossary_colon(self, text: str) -> List[Dict]:
        chunks = []
        for line in text.split("\n"):
            line  = line.strip()
            match = re.match(r'^([^:]{3,60}):\s+(.+)', line)
            if match:
                chunks.append(self._make_glossary_chunk(match.group(1), [match.group(2)]))
        return chunks

    def _chunk_glossary_inline(self, text: str) -> List[Dict]:
        text = re.sub(
            r'([A-ZÀ-Ÿa-zà-ÿ/\-()\''']{3,})([A-ZÀ-Ÿ][a-zà-ÿ])',
            lambda m: m.group(1) + "\n" + m.group(2),
            text
        )
        return self._chunk_glossary_newline(text)

    def _chunk_sectioned(self, text: str, chunk_size: int = 800) -> List[Dict]:
        pattern  = re.compile(
            r'(?m)(?=^(?:›\s+|\#{1,3}\s+|[A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ][A-ZÀÂÉÈÊËÎÏÔÙÛÜÇ\s]{5,}$))'
        )
        sections = pattern.split(text)
        chunks   = []
        for section in sections:
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
        """
        Découpe le texte en chunks par phrases avec overlap en MOTS.
        overlap=80 → les 80 derniers mots du chunk précédent sont répétés
        au début du suivant, ce qui évite de couper les mots en plein milieu.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""

        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size and current:
                chunks.append(current.strip())
                # Overlap en MOTS (pas en caractères) pour éviter les coupures
                last_words  = current.split()
                overlap_words = last_words[-overlap:] if len(last_words) > overlap else last_words
                current = " ".join(overlap_words) + " " + sentence
            else:
                current += " " + sentence

        if current.strip():
            chunks.append(current.strip())
        return chunks

    # =========================================================
    # HELPERS
    # =========================================================

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

    def _make_glossary_chunk(self, term: str, definition: List[str]) -> Dict:
        # type='narrative' et non 'glossaire' — les chunks PDF ne sont pas
        # des entrées de glossaire même si leur format ressemble à un glossaire
        return {
            "content":  f"{term}\n{' '.join(definition)}",
            "metadata": {"term": term, "type": "narrative"},
        }