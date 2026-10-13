"""jagged_demo.py -- illustrate the tokenization root cause of 'simple' LLM errors (Module Q10).
Pure stdlib. Shows (1) a naive character count, (2) a crude 'sub-word chunking' that hides letters,
and (3) how 'spelling out' (reframing the input) restores the correct count.
NOTE: this SIMULATES why the error happens at the representation level; it does not call an LLM.
"""
def naive_count(word, ch): return word.count(ch)

def chunk_as_subwords(word, size=3):
    """Pretend the tokenizer splits the word into opaque chunks (no per-letter view)."""
    return [word[i:i+size] for i in range(0, len(word), size)]

def count_from_chunks_blind(chunks, ch):
    """A model that only 'knows' chunk identities, not their letters, cannot count reliably.
    We simulate this by only counting in chunks that happen to START with the char (lossy)."""
    return sum(1 for c in chunks if c.startswith(ch))

def count_spelled_out(word, ch):
    """Reframing: spell the word out letter-by-letter, THEN count -- exposes sub-token structure."""
    spelled = "-".join(list(word))
    return spelled.split("-").count(ch)

if __name__ == "__main__":
    for word in ("strawberry", "banana"):
        ch = "r" if word == "strawberry" else "a"
        chunks = chunk_as_subwords(word)
        print(f"word={word!r}  counting {ch!r}")
        print(f"  true count (char-level)        : {naive_count(word, ch)}")
        print(f"  subword chunks (what model sees): {chunks}")
        print(f"  blind count from chunks (buggy) : {count_from_chunks_blind(chunks, ch)}")
        print(f"  after 'spell it out' reframing  : {count_spelled_out(word, ch)}")
        print()
    print("Lesson: the error lives in the REPRESENTATION (sub-word tokens hide letters);")
    print("reframing the input to expose letters fixes it with no change to the model.")
