from __future__ import annotations

import math
import re
from typing import List, Dict, Any

class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Tách câu bằng regex dựa trên dấu kết thúc câu (. ! ? .\n)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\.\n', text) if s.strip()]
        chunks = []
        
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i:i + self.max_sentences_per_chunk])
            chunks.append(chunk)
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            return [current_text[:self.chunk_size]]
        
        separator = remaining_separators[0]
        if separator == "":
            parts = list(current_text)
        else:
            parts = current_text.split(separator)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for part in parts:
            part_len = len(part)
            
            if current_len + part_len <= self.chunk_size:
                current_chunk.append(part)
                current_len += part_len + len(separator)
            else:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                
                if part_len > self.chunk_size:
                    chunks.extend(self._split(part, remaining_separators[1:]))
                else:
                    current_chunk = [part]
                    current_len = part_len
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class ParentChildChunker:
    """
    Custom Parent-Child chunking strategy specifically designed for Shopee Policies.
    It identifies major policy sections (Parents) and splits them into smaller, 
    embeddable sentences or table rows (Children) while preserving the parent context.
    """
    def __init__(self, max_sentences_per_child: int = 2):
        self.max_sentences_per_child = max_sentences_per_child

    def chunk_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into children and binds them with their parent metadata.
        Returns a list of dictionaries containing 'content' (child) and 'metadata' (parent info).
        """
        # Regex to detect section headers like "1.1. Đối Tượng Áp Dụng" or "***4.1. Đối tượng...***"
        # Matches start of line, optional asterisks, numbers separated by dots.
        pattern = re.compile(r'^(?:\*\*\*)?\d+\.\d+.*$', re.MULTILINE)
        
        matches = list(pattern.finditer(text))
        chunks = []
        
        # Handle cases where no clear sections are found
        if not matches:
            return self._process_parent("General Policy", text)

        # Process any introductory text before the first section
        if matches[0].start() > 0:
            intro_text = text[:matches[0].start()].strip()
            if intro_text:
                chunks.extend(self._process_parent("Introduction", intro_text))

        # Process each detected parent section
        for i in range(len(matches)):
            start_idx = matches[i].start()
            # The section ends where the next one begins, or at the end of the text
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            parent_block = text[start_idx:end_idx].strip()
            # Clean up the title (remove markdown bold/italic asterisks)
            parent_title = matches[i].group().strip('* ')
            
            chunks.extend(self._process_parent(parent_title, parent_block))
            
        return chunks

    def _process_parent(self, parent_title: str, parent_text: str) -> List[Dict[str, Any]]:
        """
        Internal helper to split a Parent block into Child chunks based on its structure (Table vs Text).
        """
        children = []
        
        # 1. SPECIAL CASE: Markdown Tables
        if "|" in parent_text and "\n|---" in parent_text.replace(" ", ""):
            lines = parent_text.split('\n')
            headers = []
            for line in lines:
                if line.strip().startswith('|'):
                    if '---' in line:
                        continue # Skip the separator line
                    
                    # Extract cell values, ignoring empty strings from split
                    cells = [c.strip() for c in line.split('|')[1:-1]]
                    
                    if not headers:
                        headers = cells # The first valid row is the header
                    else:
                        # Combine column headers with cell values to create a meaningful child sentence
                        child_text = ". ".join(
                            [f"{headers[i]}: {cells[i]}" for i in range(min(len(headers), len(cells))) if cells[i]]
                        )
                        if child_text:
                            children.append(child_text)
                            
        # 2. NORMAL CASE: Regular Text
        else:
            # Fallback to standard sentence splitting using regex
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\.\n', parent_text) if s.strip()]
            
            for i in range(0, len(sentences), self.max_sentences_per_child):
                child_text = " ".join(sentences[i:i + self.max_sentences_per_child])
                children.append(child_text)
                
        # 3. METADATA BINDING: Wrap each child with its parent's DNA
        results = []
        for child in children:
            if child:  # Ensure no empty chunks are added
                results.append({
                    "content": child,  # This will be embedded
                    "metadata": {
                        "parent_title": parent_title,
                        "parent_content": parent_text,  # This will be given to the LLM
                        "strategy": "parent_child"
                    }
                })
        return results


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20).chunk(text),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text)
            # "parent_child": ParentChildChunker(max_sentences_per_child=2).chunk_with_metadata(text)
        }
        
        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            result[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks
            }
        return result
