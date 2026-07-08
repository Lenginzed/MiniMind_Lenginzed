from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def discover_text_files(inputs: Iterable[str]) -> List[str]:
    files: List[str] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(str(p) for p in sorted(path.rglob("*.txt")))
        else:
            matched = sorted(glob.glob(str(path)))
            files.extend(matched if matched else [str(path)])
    unique = []
    seen = set()
    for file_path in files:
        if file_path not in seen:
            unique.append(file_path)
            seen.add(file_path)
    if not unique:
        raise ValueError("no text files found")
    for file_path in unique:
        if not Path(file_path).exists():
            raise FileNotFoundError(file_path)
    return unique


class MiniTokenizer:
    """Small Byte-level BPE wrapper for Stage 2 data pipeline smoke tests."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train_from_files(
        cls,
        files: Iterable[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
    ) -> "MiniTokenizer":
        text_files = discover_text_files(files)
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train(text_files, trainer=trainer)
        bos_id = tokenizer.token_to_id(BOS_TOKEN)
        eos_id = tokenizer.token_to_id(EOS_TOKEN)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="%s $A %s" % (BOS_TOKEN, EOS_TOKEN),
            pair="%s $A %s $B %s" % (BOS_TOKEN, EOS_TOKEN, EOS_TOKEN),
            special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
        )
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str) -> "MiniTokenizer":
        return cls(Tokenizer.from_file(path))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)

    @property
    def special_token_ids(self) -> Dict[str, Optional[int]]:
        return {
            "pad_token_id": self.token_to_id(PAD_TOKEN),
            "bos_token_id": self.token_to_id(BOS_TOKEN),
            "eos_token_id": self.token_to_id(EOS_TOKEN),
            "unk_token_id": self.token_to_id(UNK_TOKEN),
        }

    def summary(self) -> Dict[str, object]:
        data: Dict[str, object] = {"vocab_size": self.vocab_size}
        data.update(self.special_token_ids)
        return data
