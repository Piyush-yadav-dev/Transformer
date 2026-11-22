"""Synthetic character tests; no external dataset or learned model is used."""

import json

import numpy as np
import pytest

from transformer.tokenization.character_tokenizer import CharacterTokenizer


def test_round_trip_and_deterministic_unicode_order():
    text = "Café!\n猫 “Hi”"
    first = CharacterTokenizer.fit([text])
    second = CharacterTokenizer.fit([text[::-1], text])
    assert first.characters == tuple(sorted(set(text)))
    assert first.fingerprint == second.fingerprint
    assert first.decode(first.encode(text)) == text
    assert first.vocab_size == len(set(text)) + 2
    assert {first.char_to_id[char]: char for char in first.characters} == first.id_to_char
    with pytest.raises(TypeError):
        first.char_to_id["z"] = 42


def test_unknown_character_maps_once_and_is_never_silently_dropped():
    tokenizer = CharacterTokenizer.fit(["abc"])
    assert tokenizer.encode("aΩ🚀") == [
        tokenizer.char_to_id["a"],
        tokenizer.unk_id,
        tokenizer.unk_id,
    ]
    assert tokenizer.decode(tokenizer.encode("aΩ🚀"), skip_eos=True) == "a<unk><unk>"
    with pytest.raises(ValueError, match="Unknown character U\\+03A9 at position 1"):
        tokenizer.encode("aΩ", unknown="error")


def test_literal_marker_strings_are_ordinary_characters():
    text = "<unk><eos>"
    tokenizer = CharacterTokenizer.fit([text])
    literal = tokenizer.encode(text)
    assert tokenizer.unk_id not in literal and tokenizer.eos_id not in literal
    assert tokenizer.decode(literal) == text
    with_eos = tokenizer.encode(text, add_eos=True)
    assert with_eos == literal + [tokenizer.eos_id]
    assert tokenizer.decode(with_eos) == text + "<eos>"
    assert tokenizer.decode(with_eos, skip_eos=True) == text
    assert tokenizer.unk_id not in tokenizer.id_to_char
    assert tokenizer.eos_id not in tokenizer.id_to_char


def test_empty_input_and_no_implicit_normalization():
    tokenizer = CharacterTokenizer.fit(["é"])
    assert tokenizer.encode("") == []
    assert tokenizer.encode("", add_eos=True) == [tokenizer.eos_id]
    assert tokenizer.decode([]) == ""
    assert tokenizer.decode([tokenizer.eos_id]) == "<eos>"
    assert tokenizer.encode("e\u0301") == [tokenizer.unk_id, tokenizer.unk_id]


@pytest.mark.parametrize("texts", [[], [""], ["", ""]])
def test_no_training_characters_cannot_fit(texts):
    with pytest.raises(ValueError, match="no characters"):
        CharacterTokenizer.fit(texts)


def test_save_load_and_overwrite_protection(tmp_path):
    tokenizer = CharacterTokenizer.fit(["abcΩ"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path, expected_fingerprint=tokenizer.fingerprint)
    assert tokenizer.to_dict() == loaded.to_dict()
    assert loaded.encode("cΩb") == tokenizer.encode("cΩb")
    with pytest.raises(FileExistsError):
        tokenizer.save(path)
    with pytest.raises(ValueError, match="Incompatible tokenizer fingerprint"):
        CharacterTokenizer.load(path, expected_fingerprint="0" * 64)


@pytest.mark.parametrize(
    "mutation", ["version", "fingerprint", "special_id", "boolean_special_id", "order"]
)
def test_corrupt_tokenizer_is_rejected(tmp_path, mutation):
    payload = CharacterTokenizer.fit(["abc"]).to_dict()
    if mutation == "version":
        payload["format_version"] = 2
    elif mutation == "fingerprint":
        payload["fingerprint"] = "0" * 64
    elif mutation == "special_id":
        payload["special_tokens"]["eos"] = 9
    elif mutation == "boolean_special_id":
        payload["special_tokens"]["unk"] = False
    else:
        payload["characters"].reverse()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        CharacterTokenizer.load(path)


@pytest.mark.parametrize("token", [-1, 999, True, 1.0, "1"])
def test_invalid_ids_are_rejected(token):
    with pytest.raises(ValueError):
        CharacterTokenizer.fit(["a"]).decode([token])


def test_numpy_integer_ids_are_supported():
    tokenizer = CharacterTokenizer.fit(["a"])
    assert tokenizer.decode(np.asarray(tokenizer.encode("a"), dtype=np.uint16)) == "a"
