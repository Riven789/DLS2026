"""
tokenizer.py -- two tiny tokenizers for the tutorial.

  CharTokenizer : one token per character (simplest possible vocabulary)
  BPETokenizer  : byte-level Byte-Pair Encoding, the scheme used by GPT-2-style
                  models (Sennrich et al. 2016; Radford et al. 2019)

Both expose the same interface:  encode(str) -> list[int],  decode(list[int]) -> str,
vocab_size, save(path), and a classmethod load(path).
"""
from __future__ import annotations

import json
import re
from collections import Counter


class CharTokenizer:
    """Vocabulary = the set of characters seen in the training text."""

    kind = "char"

    def __init__(self, chars: list[str]):
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def token_str(self, i: int) -> str:
        return self.itos[i]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"kind": self.kind, "chars": self.itos}, f)


# GPT-2-style pre-tokenization: split text into "words" that keep their leading
# space.  BPE merges never cross these boundaries, which keeps training fast
# (we count each distinct word once) and keeps tokens linguistically sensible.
_PRETOKENIZE = re.compile(r"\s?[A-Za-z]+|\s?[0-9]+|\s?[^\sA-Za-z0-9]+|\s+")


class BPETokenizer:
    """
    Byte-level BPE.

    Start with 256 tokens, one per byte value, so *any* string can be encoded.
    Training repeats:   (1) count every adjacent pair of tokens,
                        (2) merge the most frequent pair into one new token.
    Each merge adds one entry to the vocabulary; we stop at `vocab_size`.
    Encoding replays the learned merges in the order they were learned.
    """

    kind = "bpe"

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}  # pair -> priority
        self.vocab = {i: bytes([i]) for i in range(256)}             # id -> bytes
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------ training
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256
        # each distinct word -> (list of token ids, how many times it occurs)
        word_counts = Counter(_PRETOKENIZE.findall(text))
        words = [(list(w.encode("utf-8")), c) for w, c in word_counts.items()]

        merges: list[tuple[int, int]] = []
        for new_id in range(256, vocab_size):
            pair_counts: Counter = Counter()
            for ids, c in words:
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += c
            if not pair_counts:
                break
            # most frequent pair; ties broken by the smaller pair for determinism
            best = max(pair_counts, key=lambda p: (pair_counts[p], (-p[0], -p[1])))
            merges.append(best)
            words = [(_merge(ids, best, new_id), c) for ids, c in words]
            if verbose:
                tok = cls(merges)
                print(f"merge {len(merges):4d}: {tok.token_str(best[0])!r} + "
                      f"{tok.token_str(best[1])!r} -> {tok.token_str(new_id)!r}"
                      f"   (count {pair_counts[best]})")
        return cls(merges)

    # ------------------------------------------------------------ inference
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _encode_word(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge that is present in this word
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = _merge(ids, pair, 256 + self.rank[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in _PRETOKENIZE.findall(text):
            out.extend(self._encode_word(word))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        return self.vocab[i].decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"kind": self.kind, "merges": self.merges}, f)


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def load_tokenizer(path: str):
    with open(path) as f:
        d = json.load(f)
    if d["kind"] == "char":
        return CharTokenizer(d["chars"])
    return BPETokenizer([tuple(m) for m in d["merges"]])


if __name__ == "__main__":
    # The worked example from the slides: learn 6 merges on a toy corpus.
    corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
    tok = BPETokenizer.train(corpus, 256 + 6, verbose=True)
    for s in ["lowest", "newer"]:
        ids = tok.encode(s)
        print(f"{s!r} -> {[tok.token_str(i) for i in ids]} -> {ids}")
        assert tok.decode(ids) == s
