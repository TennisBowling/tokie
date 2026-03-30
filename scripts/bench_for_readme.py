"""Benchmark tokie vs HuggingFace and tiktoken for README tables.

Run: python scripts/bench_for_readme.py

Measures median of 10 runs for single-string encode at 45KB and 900KB.
"""

import json
import time

import tokie
from tokenizers import Tokenizer as HFTokenizer

SHORT = "The quick brown fox jumps over the lazy dog. " * 10   # ~450B
TEXT_45K = SHORT * 100   # ~45 KB
TEXT_900K = SHORT * 2000 # ~900 KB


def bench_encode(tokenizer, text, warmup=3, iters=10):
    for _ in range(warmup):
        tokenizer.encode(text)
    times = []
    tokens = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        enc = tokenizer.encode(text)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        tokens = len(enc.ids)
    median = sorted(times)[len(times) // 2]
    return median * 1000, tokens


def bench_hf_encode(hf, text, warmup=3, iters=10):
    for _ in range(warmup):
        hf.encode(text)
    times = []
    tokens = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        enc = hf.encode(text)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        tokens = len(enc.ids)
    median = sorted(times)[len(times) // 2]
    return median * 1000, tokens


def bench_tiktoken(bpe, text, warmup=3, iters=10):
    for _ in range(warmup):
        bpe.encode(text)
    times = []
    tokens = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        ids = bpe.encode(text)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        tokens = len(ids)
    median = sorted(times)[len(times) // 2]
    return median * 1000, tokens


def bench_batch(tokenizer, texts, warmup=2, iters=5):
    for _ in range(warmup):
        tokenizer.encode_batch(texts)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tokenizer.encode_batch(texts)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[len(times) // 2] * 1000


def bench_hf_batch(hf, texts, warmup=2, iters=5):
    for _ in range(warmup):
        hf.encode_batch(texts)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        hf.encode_batch(texts)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[len(times) // 2] * 1000


def bench_loading(repo_id, warmup=2, iters=5):
    # tokie from HF (downloads tokenizer.json, converts)
    for _ in range(warmup):
        tokie.Tokenizer.from_pretrained(repo_id)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tokie.Tokenizer.from_pretrained(repo_id)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    tokie_ms = sorted(times)[len(times) // 2] * 1000

    # HF
    for _ in range(warmup):
        HFTokenizer.from_pretrained(repo_id)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        HFTokenizer.from_pretrained(repo_id)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    hf_ms = sorted(times)[len(times) // 2] * 1000

    return tokie_ms, hf_ms


def main():
    # Models for main table (same as README)
    models_vs_hf = [
        ("BERT",       "bert-base-uncased",              "bert-base-uncased"),
        ("GPT-2",      "openai-community/gpt2",          "openai-community/gpt2"),
        ("Llama 3",    "meta-llama/Llama-3.2-1B",        "meta-llama/Llama-3.2-1B"),
        ("Qwen 3",     "Qwen/Qwen3-0.6B",               "Qwen/Qwen3-0.6B"),
        ("ModernBERT", "answerdotai/ModernBERT-base",    "answerdotai/ModernBERT-base"),
        ("Gemma 3",    "google/gemma-3-4b-it",           "google/gemma-3-4b-it"),
    ]

    print("=" * 90)
    print("tokie vs HuggingFace tokenizers — single string encode")
    print("=" * 90)
    print()
    print(f"{'Model':<15} {'Size':<10} {'tokie':>10} {'HF':>10} {'vs HF':>8}")
    print("-" * 55)

    readme_rows = []

    for name, tokie_repo, hf_repo in models_vs_hf:
        tok = tokie.Tokenizer.from_pretrained(tokie_repo)
        hf = HFTokenizer.from_pretrained(hf_repo)

        for size_name, text in [("45 KB", TEXT_45K), ("900 KB", TEXT_900K)]:
            tokie_ms, tokie_tokens = bench_encode(tok, text)
            hf_ms, hf_tokens = bench_hf_encode(hf, text)
            speedup = hf_ms / tokie_ms

            print(f"{name:<15} {size_name:<10} {tokie_ms:>8.2f}ms {hf_ms:>8.1f}ms {speedup:>6.0f}x")

            readme_rows.append({
                "model": name, "size": size_name,
                "tokie_ms": round(tokie_ms, 2), "hf_ms": round(hf_ms, 1),
                "speedup": round(speedup), "tokens": tokie_tokens,
            })

    # tiktoken comparison
    print()
    print("=" * 90)
    print("tokie vs tiktoken — single string encode")
    print("=" * 90)
    print()
    print(f"{'Model':<20} {'Size':<10} {'tokie':>10} {'tiktoken':>10} {'Speedup':>8}")
    print("-" * 60)

    import tiktoken

    tiktoken_rows = []

    for model_name, tiktoken_enc in [("cl100k (GPT-4)", "cl100k_base"), ("o200k (GPT-4o)", "o200k_base")]:
        enc_name = "cl100k_base" if "cl100k" in tiktoken_enc else "o200k_base"
        bpe = tiktoken.get_encoding(enc_name)
        tokie_repo = "tokiers/cl100k" if "cl100k" in tiktoken_enc else "tokiers/o200k"
        tok = tokie.Tokenizer.from_pretrained(tokie_repo)

        for size_name, text in [("45 KB", TEXT_45K), ("900 KB", TEXT_900K)]:
            tokie_ms, _ = bench_encode(tok, text)
            tk_ms, _ = bench_tiktoken(bpe, text)
            speedup = tk_ms / tokie_ms

            print(f"{model_name:<20} {size_name:<10} {tokie_ms:>8.2f}ms {tk_ms:>8.2f}ms {speedup:>6.1f}x")

            tiktoken_rows.append({
                "model": model_name, "size": size_name,
                "tokie_ms": round(tokie_ms, 2), "tiktoken_ms": round(tk_ms, 2),
                "speedup": round(speedup, 1),
            })

    # Batch benchmarks
    print()
    print("=" * 90)
    print("Batch encode (100 x 45KB strings)")
    print("=" * 90)
    print()
    print(f"{'Model':<15} {'tokie':>10} {'HF':>10} {'vs HF':>8}")
    print("-" * 45)

    batch_texts = [TEXT_45K] * 100

    for name, tokie_repo, hf_repo in models_vs_hf[:5]:
        tok = tokie.Tokenizer.from_pretrained(tokie_repo)
        hf = HFTokenizer.from_pretrained(hf_repo)

        tokie_ms = bench_batch(tok, batch_texts)
        hf_ms = bench_hf_batch(hf, batch_texts)
        speedup = hf_ms / tokie_ms

        print(f"{name:<15} {tokie_ms:>8.0f}ms {hf_ms:>8.0f}ms {speedup:>6.1f}x")

    # Loading benchmarks
    print()
    print("=" * 90)
    print("Loading time (from_pretrained, cached on disk)")
    print("=" * 90)
    print()

    loading_models = [
        ("BERT", "bert-base-uncased"),
        ("GPT-2", "openai-community/gpt2"),
        ("Llama 3.2", "meta-llama/Llama-3.2-1B"),
        ("cl100k", "tokiers/cl100k"),
        ("o200k", "tokiers/o200k"),
    ]

    print(f"{'Model':<15} {'tokie':>10} {'HF':>10} {'Speedup':>8}")
    print("-" * 45)

    loading_rows = []

    for name, repo in loading_models:
        tokie_ms, hf_ms = bench_loading(repo)
        speedup = hf_ms / tokie_ms
        print(f"{name:<15} {tokie_ms:>8.1f}ms {hf_ms:>8.1f}ms {speedup:>6.1f}x")
        loading_rows.append({
            "model": name, "tokie_ms": round(tokie_ms, 1),
            "hf_ms": round(hf_ms, 1), "speedup": round(speedup, 1),
        })

    # Save all results
    all_results = {
        "vs_hf": readme_rows,
        "vs_tiktoken": tiktoken_rows,
        "loading": loading_rows,
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
