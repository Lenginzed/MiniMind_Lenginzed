from __future__ import annotations

from pathlib import Path

from minillm.tokenizer import MiniTokenizer


def test_toy_tokenizer_train_load_encode_decode(tmp_path: Path) -> None:
    corpus = tmp_path / "toy.txt"
    corpus.write_text(
        "\n".join(
            [
                "Mini language models test tokenizers.",
                "小模型 tokenizer smoke test.",
                "RoPE and GQA are transformer components.",
            ]
            * 20
        ),
        encoding="utf-8",
    )
    out = tmp_path / "tok.json"
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer.save(str(out))
    loaded = MiniTokenizer.load(str(out))
    ids = loaded.encode("Mini tokenizer 小模型")
    decoded = loaded.decode(ids)
    assert out.exists()
    assert loaded.vocab_size <= 128
    assert len(ids) > 0
    assert isinstance(decoded, str)
    specials = loaded.special_token_ids
    assert specials["pad_token_id"] is not None
    assert specials["bos_token_id"] is not None
    assert specials["eos_token_id"] is not None
    assert specials["unk_token_id"] is not None
