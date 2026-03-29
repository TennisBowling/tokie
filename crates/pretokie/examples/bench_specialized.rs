//! Benchmark pretokenizer throughput on enwik8.
//!
//! Usage: cargo run -p pretokie --example bench_specialized --release

use pretokie::Gpt2;
use pretokie::Cl100k;
use std::time::Instant;

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    let iters = 20;

    // GPT-2
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..iters {
            count = Gpt2::new(&text).count();
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("GPT-2:  {throughput:>8.1} MB/s  ({count} pieces)");
    }

    // CL100K
    {
        let _ = Cl100k::new(&text).count();
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..iters {
            count = Cl100k::new(&text).count();
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("CL100K: {throughput:>8.1} MB/s  ({count} pieces)");
    }
}
