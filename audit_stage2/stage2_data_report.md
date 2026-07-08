# Stage 2 Data Report

## Toy Corpus

- Path: `data/raw/toy_corpus.txt`
- Lines: `6000`
- Bytes: `445667`
- Purpose: local pipeline validation only; this is not a real pretraining dataset.

## Tokenizer

- Type: Byte-level BPE via Hugging Face `tokenizers`
- Path: `data/tokenizers/toy_tokenizer.json`
- Vocab size: `1000`
- Special tokens:
  - pad_token_id: `0`
  - bos_token_id: `1`
  - eos_token_id: `2`
  - unk_token_id: `3`
- Sample decode: `Mini LLM smoke test: 小模型检查 tokenizer encode and decode.`

Byte-level BPE was chosen because the toy corpus mixes English, Chinese, punctuation, numbers, and technical tokens. It is robust for arbitrary UTF-8 text and avoids adding a separate language-specific segmenter at this stage.

## Tokenized Dataset

- Total tokens: `86578`
- Train tokens: `77921`
- Val tokens: `8657`
- Block size: `64`
- Train samples: `1217`
- Val samples: `135`
- Train path: `data\processed\train.npy`
- Val path: `data\processed\val.npy`
