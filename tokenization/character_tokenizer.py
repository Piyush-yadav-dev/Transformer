"""A frozen Unicode character vocabulary with distinct unknown and story-end IDs."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Literal

from transformer.data.preparation import canonical_json, fingerprint, write_json


@dataclass(frozen=True, slots=True)
class CharacterTokenizer:
    characters: tuple[str, ...]
    _char_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _id_to_char: Mapping[int, str] = field(init=False, repr=False, compare=False)
    format_version: ClassVar[int] = 1
    unk_id: ClassVar[int] = 0
    eos_id: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.characters, tuple) or not self.characters:
            raise ValueError("Tokenizer characters must be a non-empty tuple.")
        if any(
            not isinstance(char, str) or len(char) != 1 or 0xD800 <= ord(char) <= 0xDFFF
            for char in self.characters
        ):
            raise ValueError("Every vocabulary entry must be one valid Unicode scalar character.")
        if tuple(sorted(set(self.characters))) != self.characters:
            raise ValueError(
                "Vocabulary characters must be unique and in Unicode code-point order."
            )
        mapping = {char: index + 2 for index, char in enumerate(self.characters)}
        object.__setattr__(self, "_char_to_id", MappingProxyType(mapping))
        object.__setattr__(
            self, "_id_to_char", MappingProxyType({v: k for k, v in mapping.items()})
        )

    @property
    def char_to_id(self) -> Mapping[str, int]:
        return self._char_to_id

    @property
    def id_to_char(self) -> Mapping[int, str]:
        """Character IDs only; special IDs are deliberately outside this mapping."""
        return self._id_to_char

    @property
    def vocab_size(self) -> int:
        return len(self.characters) + 2

    @classmethod
    def fit(cls, texts: Iterable[str]) -> "CharacterTokenizer":
        characters: set[str] = set()
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("Tokenizer fitting requires strings.")
            characters.update(text)
        if not characters:
            raise ValueError("Cannot fit a tokenizer: the allowed training text has no characters.")
        return cls(tuple(sorted(characters)))

    def encode(
        self, text: str, *, add_eos: bool = False, unknown: Literal["replace", "error"] = "replace"
    ) -> list[int]:
        if not isinstance(text, str):
            raise ValueError("Tokenizer input must be a string.")
        if type(add_eos) is not bool or unknown not in ("replace", "error"):
            raise ValueError("add_eos must be boolean and unknown must be 'replace' or 'error'.")
        result = []
        for position, char in enumerate(text):
            token = self._char_to_id.get(char)
            if token is None:
                if unknown == "error":
                    raise ValueError(f"Unknown character U+{ord(char):04X} at position {position}.")
                token = self.unk_id
            result.append(token)
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(self, ids: Iterable[int], *, skip_eos: bool = False) -> str:
        if type(skip_eos) is not bool:
            raise ValueError("skip_eos must be boolean.")
        result = []
        for token in ids:
            if isinstance(token, bool) or not isinstance(token, Integral):
                raise ValueError(
                    "Token IDs must be integers, not booleans or floating-point values."
                )
            token = int(token)
            if not 0 <= token < self.vocab_size:
                raise ValueError(f"Token ID {token} is outside vocabulary size {self.vocab_size}.")
            if token == self.unk_id:
                result.append("<unk>")
            elif token == self.eos_id:
                if not skip_eos:
                    result.append("<eos>")
            else:
                result.append(self._id_to_char[token])
        return "".join(result)

    def _payload(self) -> dict:
        return {
            "format_version": self.format_version,
            "kind": "character",
            "characters": list(self.characters),
            "special_tokens": {"unk": self.unk_id, "eos": self.eos_id},
            "normalization": "none",
            "unknown_policy": "one_unk_per_unknown_character",
            "decode_labels": {"unk": "<unk>", "eos": "<eos>"},
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self._payload())

    def to_dict(self) -> dict:
        return {**self._payload(), "fingerprint": self.fingerprint}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path, *, expected_fingerprint: str | None = None) -> "CharacterTokenizer":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or type(payload.get("format_version")) is not int:
                raise ValueError("Missing integer tokenizer format version.")
            if payload["format_version"] != cls.format_version:
                raise ValueError(
                    f"Unsupported tokenizer format version: {payload['format_version']}."
                )
            if not isinstance(payload.get("characters"), list):
                raise ValueError("Tokenizer characters must be a JSON list.")
            tokenizer = cls(tuple(payload["characters"]))
            if canonical_json(payload) != canonical_json(tokenizer.to_dict()):
                raise ValueError("Tokenizer metadata or fingerprint is inconsistent.")
            if expected_fingerprint is not None and tokenizer.fingerprint != expected_fingerprint:
                raise ValueError("Incompatible tokenizer fingerprint.")
            return tokenizer
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed tokenizer file {path}: {exc}.") from exc
