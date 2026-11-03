from model.tokenizer import CharTokenizer


def test_round_trip():
    text = "le chat dort"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size_counts_specials():
    tok = CharTokenizer.from_text("abcab")
    assert tok.vocab_size == 3 + 2  # a, b, c + <pad>, <unk>


def test_special_ids_are_stable():
    tok = CharTokenizer.from_text("xyz")
    assert tok.pad_id == 0
    assert tok.unk_id == 1


def test_unknown_char_maps_to_unk():
    tok = CharTokenizer.from_text("abc")
    ids = tok.encode("abz")
    assert ids[-1] == tok.unk_id


def test_decode_skips_special_tokens():
    tok = CharTokenizer.from_text("abc")
    assert tok.decode(tok.encode("abz")) == "ab"


def test_save_and_load(tmp_path):
    tok = CharTokenizer.from_text("héllo wörld")
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.itos == tok.itos
    assert loaded.encode("hello") == tok.encode("hello")
