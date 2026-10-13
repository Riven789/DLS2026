"""An exactly reversible UTF-8 byte tokenizer; no vocabulary download."""
class ByteTokenizer:
    vocab_size = 258
    bos_id, eos_id = 256, 257

    def encode(self, text, bos=False, eos=False):
        return ([self.bos_id] if bos else []) + list(text.encode('utf-8')) + ([self.eos_id] if eos else [])

    def decode(self, ids):
        # Generated arbitrary bytes may not form valid UTF-8. Replacement characters
        # make that visible. Decode the complete byte stream, not one byte at a time.
        return bytes(i for i in ids if 0 <= i < 256).decode('utf-8', errors='replace')


def toy_bpe_demo():
    """A hand-specified, ranked merge list to explain BPE (not a trained tokenizer)."""
    pieces = list('lower')
    trace = [pieces.copy()]
    for left, right in [('l', 'o'), ('lo', 'w'), ('e', 'r')]:
        merged, i = [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and pieces[i:i+2] == [left, right]:
                merged.append(left + right)
                i += 2
            else:
                merged.append(pieces[i]); i += 1
        pieces = merged
        trace.append(pieces.copy())
    return trace
