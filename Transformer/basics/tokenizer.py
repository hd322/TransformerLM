# cs336_basics/tokenizer.py
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Add special tokens to vocab if missing
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.vocab_inv:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = st_bytes
                self.vocab_inv[st_bytes] = new_id

        # merge rank: (bytes1, bytes2) -> priority index
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        # sort special tokens longest-first so greedy regex matches <|endoftext|><|endoftext|> before <|endoftext|>
        self._sorted_special = sorted(self.special_tokens, key=len, reverse=True)
        if self._sorted_special:
            pattern = "(" + "|".join(re.escape(s) for s in self._sorted_special) + ")"
            self._special_re = re.compile(pattern)
        else:
            self._special_re = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json
        with open(vocab_filepath) as f:
            raw = json.load(f)
        vocab = {int(k): bytes(v) for k, v in raw.items()}
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    a, b = line.split(" ", 1)
                    merges.append((a.encode(), b.encode()))
        return cls(vocab, merges, special_tokens)

    def _bpe(self, word_bytes: bytes) -> list[int]:
        """Apply BPE merges to a single pre-token (bytes)."""
        tokens = [bytes([b]) for b in word_bytes]
        while len(tokens) > 1:
            # find the applicable merge with the lowest rank
            best_rank, best_i = None, None
            for i in range(len(tokens) - 1):
                rank = self.merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i is None:
                break
            merged = tokens[best_i] + tokens[best_i + 1]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2:]
        return [self.vocab_inv[t] for t in tokens]

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a plain text chunk (no special tokens inside)."""
        ids = []
        for m in re.finditer(PAT, text):
            ids.extend(self._bpe(m.group().encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if not self._special_re:
            return self._encode_chunk(text)
        ids = []
        for part in self._special_re.split(text):
            if part in self.special_tokens:
                ids.append(self.vocab_inv[part.encode("utf-8")])
            elif part:
                ids.extend(self._encode_chunk(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Memory-efficient: process one line at a time."""
        if not self._special_re:
            # no special tokens: each line is independent
            for line in iterable:
                yield from self._encode_chunk(line)
            return

        # with special tokens: buffer only what's needed to detect them
        buffer = ""
        for line in iterable:
            buffer += line
            parts = self._special_re.split(buffer)
            # last part may be an incomplete special token prefix — hold it back
            for part in parts[:-1]:
                if part in self.special_tokens:
                    yield self.vocab_inv[part.encode("utf-8")]
                elif part:
                    yield from self._encode_chunk(part)
            buffer = parts[-1]
        # flush remainder
        if buffer:
            for part in self._special_re.split(buffer):
                if part in self.special_tokens:
                    yield self.vocab_inv[part.encode("utf-8")]
                elif part:
                    yield from self._encode_chunk(part)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")