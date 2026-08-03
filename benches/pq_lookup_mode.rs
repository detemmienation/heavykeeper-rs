//! Benchmark comparing hashmap vs linear-pq lookup strategies.
//!
//! Run both modes and compare:
//!   cargo bench --bench pq_lookup_mode              # hashmap (default)
//!   cargo bench --bench pq_lookup_mode --features linear-pq  # linear scan
//!
//! Workloads:
//! - **pathological**: only K distinct items, every add hits the PQ lookup.
//! - **realistic**: large universe, most items never enter top-K.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use heavykeeper::CuckooTopK;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Zipf};

const WIDTH: usize = 4096;
const DEPTH: usize = 4;
const DECAY: f64 = 0.9;
const SEED: u64 = 0xBEEF;
const N: usize = 500_000;

/// Pathological: only K distinct items cycle, so every `add` hits the PQ.
fn gen_pathological(k: usize, n: usize) -> Vec<u64> {
    (0..n).map(|i| (i % k) as u64).collect()
}

/// Realistic: Zipf(1.2) over 1M universe — only ~K items accumulate enough
/// count to live in the PQ; the vast majority miss.
fn gen_realistic(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = Zipf::new(1_000_000.0, 1.2).expect("valid zipf");
    (0..n).map(|_| dist.sample(&mut rng) as u64).collect()
}

fn bench_pathological(c: &mut Criterion) {
    let mode = if cfg!(feature = "linear-pq") {
        "linear"
    } else {
        "hashmap"
    };

    let mut group = c.benchmark_group(format!("pathological_{mode}"));
    group.sample_size(30);
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(8));

    for &k in &[10usize, 50, 100, 500] {
        let data = gen_pathological(k, N);
        group.throughput(criterion::Throughput::Elements(N as u64));
        group.bench_with_input(BenchmarkId::new("CuckooTopK", k), &data, |b, data| {
            b.iter_with_setup(
                || CuckooTopK::<u64>::with_seed(k, WIDTH, DEPTH, DECAY, SEED),
                |mut topk| {
                    for item in data.iter() {
                        topk.add(black_box(item), 1);
                    }
                },
            );
        });
    }
    group.finish();
}

fn bench_realistic(c: &mut Criterion) {
    let mode = if cfg!(feature = "linear-pq") {
        "linear"
    } else {
        "hashmap"
    };

    let mut group = c.benchmark_group(format!("realistic_{mode}"));
    group.sample_size(30);
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(8));

    let data = gen_realistic(N);

    for &k in &[10usize, 50, 100, 500] {
        group.throughput(criterion::Throughput::Elements(N as u64));
        group.bench_with_input(BenchmarkId::new("CuckooTopK", k), &data, |b, data| {
            b.iter_with_setup(
                || CuckooTopK::<u64>::with_seed(k, WIDTH, DEPTH, DECAY, SEED),
                |mut topk| {
                    for item in data.iter() {
                        topk.add(black_box(item), 1);
                    }
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pathological, bench_realistic);
criterion_main!(benches);
