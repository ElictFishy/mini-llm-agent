import json
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"
SPECIAL_TOKENS = [PAD, UNK]


class CharTokenizer:
    """Character-level tokenizer: one character = one token."""

    def __init__(self, chars):
        # Special tokens come first so their ids stay stable (pad=0, unk=1).
        self.itos = SPECIAL_TOKENS + sorted(set(chars) - set(SPECIAL_TOKENS))
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def unk_id(self):
        return self.stoi[UNK]

    def encode(self, text):
        # Characters missing from the vocabulary fall back to <unk>.
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids, skip_special=True):
        toks = [self.itos[i] for i in ids]
        if skip_special:
            toks = [t for t in toks if t not in SPECIAL_TOKENS]
        return "".join(toks)

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls("")
        tok.itos = itos
        tok.stoi = {t: i for i, t in enumerate(itos)}
        return tok
