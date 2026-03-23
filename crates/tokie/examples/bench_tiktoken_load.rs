//! Benchmark tiktoken-rs loading/construction time
//!
//! Run with: cargo run --example bench_tiktoken_load --release

use std::time::Instant;

fn main() {
    let iters = 3;

    // r50k (GPT-2)
    // Warmup
    let _ = tiktoken_rs::r50k_base().unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = tiktoken_rs::r50k_base().unwrap();
    }
    let elapsed = start.elapsed() / iters as u32;
    println!("r50k (GPT-2):    {:.2} ms", elapsed.as_secs_f64() * 1000.0);

    // cl100k (GPT-4)
    let _ = tiktoken_rs::cl100k_base().unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = tiktoken_rs::cl100k_base().unwrap();
    }
    let elapsed = start.elapsed() / iters as u32;
    println!("cl100k (GPT-4):  {:.2} ms", elapsed.as_secs_f64() * 1000.0);

    // o200k (GPT-4o)
    let _ = tiktoken_rs::o200k_base().unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = tiktoken_rs::o200k_base().unwrap();
    }
    let elapsed = start.elapsed() / iters as u32;
    println!("o200k (GPT-4o):  {:.2} ms", elapsed.as_secs_f64() * 1000.0);
}
