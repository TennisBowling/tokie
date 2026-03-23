<p align="center">
  <img src="https://raw.githubusercontent.com/chonkie-inc/tokie/main/assets/tokie-banner.png" alt="tokie">
</p>

<p align="center">
  <b>50x faster tokenization, 10x smaller model files, 100% accurate</b>
</p>

<p align="center">
  <a href="https://github.com/chonkie-inc/tokie">GitHub</a> · <a href="https://crates.io/crates/tokie">crates.io</a> · <a href="https://huggingface.co/tokiers">HuggingFace</a>
</p>

---

**tokie** is a fast, correct tokenizer library built in Rust with Python bindings. It supports BPE (GPT-2, tiktoken, SentencePiece), WordPiece (BERT), and Unigram encoders.

## Installation

```bash
pip install tokie
```

## Quick Start

```python
import tokie

# Load from HuggingFace Hub (tries .tkz first, falls back to tokenizer.json)
tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")

# Encode
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [101, 7592, 1010, 2088, 999, 102]

# Decode
text = tokenizer.decode(tokens)

# Token count
count = tokenizer.count_tokens("Hello, world!")

# Vocabulary size
print(tokenizer.vocab_size)  # 30522
```

## Pair Encoding (Cross-Encoders)

```python
pair = tokenizer.encode_pair("How are you?", "I am fine.")
print(pair.ids)             # [101, 2129, 2024, 2017, 1029, 102, 1045, 2572, 2986, 1012, 102]
print(pair.attention_mask)  # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(pair.type_ids)        # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```

## Save & Load (.tkz format)

tokie's binary `.tkz` format is ~10x smaller than `tokenizer.json` and loads in ~5ms:

```python
tokenizer.save("model.tkz")
tokenizer = tokie.Tokenizer.from_file("model.tkz")
```

## Supported Models

Works with any HuggingFace tokenizer — GPT-2, BERT, Llama 3/4, Mistral, Phi, Qwen, T5, XLM-RoBERTa, and more.

## License

MIT OR Apache-2.0
