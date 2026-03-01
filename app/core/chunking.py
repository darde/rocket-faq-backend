from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_faq_document(content: str) -> list[Chunk]:
    """Split FAQ markdown into one chunk per Q&A pair, preserving section context."""
    chunks: list[Chunk] = []
    current_section = ""
    current_subsection = ""

    lines = content.split("\n")
    i = 0
    chunk_index = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("## ") and not stripped.startswith("### "):
            current_section = stripped[3:].strip()
            current_subsection = ""
            i += 1
            continue

        if stripped.startswith("### "):
            current_subsection = stripped[4:].strip()
            i += 1
            continue

        if stripped.startswith("**Q:"):
            question_line = stripped
            answer_lines: list[str] = []
            i += 1

            while i < len(lines):
                next_stripped = lines[i].strip()
                if next_stripped.startswith("**Q:"):
                    break
                if next_stripped.startswith("## ") or (
                    next_stripped.startswith("### ")
                    and not next_stripped.startswith("###  ")
                ):
                    break
                if next_stripped == "---":
                    break
                answer_lines.append(lines[i])
                i += 1

            q_text = re.sub(r"\*\*", "", question_line)
            q_text = q_text.replace("Q:", "").strip()

            a_text = "\n".join(answer_lines).strip()
            a_text = re.sub(r"^A:\s*", "", a_text)

            chunk_text = f"Question: {q_text}\nAnswer: {a_text}"

            chunk_id = f"chunk_{chunk_index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "section": current_section,
                        "subsection": current_subsection,
                        "question": q_text,
                        "source": "rocket_mortgage_faq",
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1
            continue

        i += 1

    logger.info("chunking_complete", total_chunks=len(chunks))
    return chunks
