"""Split an LLM token stream into target-language clauses for early TTS."""
from __future__ import annotations

CLAUSE_PUNCT = "。！？!?;；，,"
SENTENCE_PUNCT = "。！？!?"


def meaningful(text: str) -> bool:
    return any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text or "")


class ClauseSplitter:
    """Emit a clause as soon as punctuation (or a length cap) is reached.

    The first clause is what starts TTS while the LLM is still generating the
    rest of the sentence.  Punctuation is preferred; a hard cap at max_chars
    keeps the first-audio latency bounded for long clauses without commas.
    """

    def __init__(self, min_chars: int = 6, max_chars: int = 26):
        self.min_chars = max(2, min_chars)
        self.max_chars = max(self.min_chars + 2, max_chars)
        self.buf = ""

    def feed(self, text: str) -> list[str]:
        if text:
            self.buf += text
        out: list[str] = []
        while True:
            idx = max((self.buf.rfind(p) for p in CLAUSE_PUNCT), default=-1)
            if idx + 1 >= self.min_chars:
                piece = self.buf[:idx + 1].strip()
                self.buf = self.buf[idx + 1:]
                if meaningful(piece):
                    out.append(piece)
                continue
            if len(self.buf) >= self.max_chars:
                # Prefer a word boundary for English, otherwise hard cut.
                cut = self.buf.rfind(" ", self.min_chars, self.max_chars)
                if cut < self.min_chars:
                    cut = self.max_chars
                piece = self.buf[:cut].strip()
                self.buf = self.buf[cut:]
                if meaningful(piece):
                    out.append(piece)
                continue
            break
        return out

    def flush(self) -> str:
        piece = self.buf.strip()
        self.buf = ""
        return piece if meaningful(piece) else ""
