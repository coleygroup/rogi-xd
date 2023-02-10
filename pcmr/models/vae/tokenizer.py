from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Iterable, Iterator, Sequence

from pcmr.utils import Configurable

_SMILES_PATTERN = r"(\[|\]|Br?|C[u,l]?|Zn|S[i,n]?|Li|Na?|Fe?|H|K|O|P|I|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@{1,2}|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
_SMILES_VOCAB = [*"HBCNOSPFIcnosp[]()123456@-+=#/\\", "Cl", "Br", "@@"]


@dataclass(eq=True, frozen=True)
class SpecialTokens:
    SOS: str = "<SOS>"
    EOS: str = "<EOS>"
    PAD: str = "<PAD>"
    UNK: str = "<UNK>"

    def __iter__(self) -> Iterator[str]:
        return iter([self.SOS, self.EOS, self.PAD, self.UNK])

    def __contains__(self, t: str) -> bool:
        """is `t` a special token?"""
        return any(t == st for st in self)


class Tokenizer(Configurable):
    def __init__(self, pattern: str, tokens: Iterable[str], st: SpecialTokens = SpecialTokens()):
        if any(t in st for t in tokens):
            raise ValueError("'tokens' and 'special_tokens' contain overlapping tokens!")

        self.pattern = re.compile(pattern)
        self.st = st

        tokens = sorted(tokens) + list(self.st)
        self.t2i = {t: i for i, t in enumerate(tokens)}
        self.i2t = {i: t for i, t in enumerate(tokens)}

    def __len__(self) -> int:
        """the number of the tokens in this tokenizer"""
        return len(self.t2i)

    def __contains__(self, t: str) -> bool:
        """is the token 't' in this tokenizer's vocabulary?"""
        return t in self.t2i

    def __call__(self, word: str) -> list[int]:
        return self.encode(word)

    @property
    def SOS(self) -> int:
        """the index of the start-of-sequence token"""
        return self.t2i[self.st.SOS]

    @property
    def EOS(self) -> int:
        """the index of the end-of-sequence token"""
        return self.t2i[self.st.EOS]

    @property
    def PAD(self) -> int:
        """the index of the pad token"""
        return self.t2i[self.st.PAD]

    @property
    def UNK(self) -> int:
        """index of the unknown symbol token"""
        return self.t2i[self.st.UNK]

    def encode(self, word: str) -> list[int]:
        """encode the input word into a list of tokens"""
        return self.tokens2ids(self.tokenize(word))

    def decode(self, ids: Sequence[int]) -> str:
        """decode the sequence of tokens to the corresponding word"""
        return "".join(self.ids2tokens(ids))

    def tokenize(self, word: str) -> list[str]:
        """tokenize the input word"""
        return list(self.pattern.findall(word))

    def tokens2ids(self, tokens: Iterable[str], add_st: bool = True) -> list[int]:
        ids = [self.t2i.get(t, self.UNK) for t in tokens]
        if add_st:
            ids = [self.SOS, *ids, self.EOS]

        return ids

    def ids2tokens(
        self, ids: Sequence[int], rem_st: bool = True, rem_eos: bool = True
    ) -> list[str]:
        if len(ids) == 0:
            return []

        if rem_st:
            sos, *ids_, eos = ids
            if sos == self.SOS and eos == self.EOS:
                ids = ids_
            elif sos == self.SOS:
                ids = [*ids_, eos]
            elif eos == self.EOS:
                ids = [sos, *ids_]

        return [self.i2t.get(i, self.st.UNK) for i in ids]

    def to_config(self) -> dict:
        return {
            "pattern": self.pattern.pattern,
            "tokens": list(t for t in self.t2i.keys() if t not in self.st),
            "st": asdict(self.st),
        }

    @classmethod
    def from_config(cls, config: dict):
        st = SpecialTokens(**config["st"])

        config = config | dict(st=st)
        return cls(**config)

    @classmethod
    def from_corpus(
        cls, pattern: str, corpus: Iterable[str], st: SpecialTokens = SpecialTokens()
    ) -> Tokenizer:
        """Build a tokenizer from the input corpus with the given tokenization scheme

        Parameters
        ----------
        pattern : str
            a regular expression defining the tokenization scheme
        corpus : Iterable[str]
            a set of words from which to build a vocabulary.
        st : SpecialTokens, default=SpecialTokens()
            the special tokens to use when building the vocabulary

        Returns
        -------
        Tokenizer
        """
        pattern = re.compile(pattern)
        vocab = [pattern.findall(word) for word in corpus]

        return cls(pattern.pattern, vocab, st)

    @classmethod
    def smiles_tokenizer(cls, st: SpecialTokens = SpecialTokens()) -> Tokenizer:
        """build a tokenizer for SMILES strings"""
        return cls(_SMILES_PATTERN, _SMILES_VOCAB, st)
