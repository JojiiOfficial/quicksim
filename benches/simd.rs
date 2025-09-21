use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use quicksim::prelude::ArrayF32SimdExt;
use quicksim::traits::array_u32::ArrayU32SimdExt;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{RngCore, SeedableRng};

const U32_ARRAY_LEN_TO_CHECK: [usize; 8] = [16, 32, 64, 150, 530, 1028, 5010, 8000];

fn find_u32(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("find_u32");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let needle = rng.next_u32();
        let data: Vec<_> = (0..size)
            .map(|_| {
                loop {
                    let r = rng.next_u32();
                    if r != needle {
                        break r;
                    }
                }
            })
            .collect();
        assert!(!data.contains(&needle));

        let mut data_avg = data.clone();
        data_avg[data.len() / 2] = needle;

        let mut data_worst = data.clone();
        data_worst[size - 1] = needle;

        g.bench_with_input(BenchmarkId::new("simd-avg", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data_avg.find_simd(needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("iter-avg", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(find_iter(&data_avg, needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("simd-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data_worst.find_simd(needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("iter-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(find_iter(&data_worst, needle));
            });
        });
    }
}

fn count_u32(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("count_u32");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let data: Vec<_> = (0..size).map(|_| rng.next_u32()).collect();
        let needle = *data.choose(&mut rng).unwrap();

        g.bench_with_input(BenchmarkId::new("simd", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data.count_simd(needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("iter", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(count_iter(&data, needle));
            });
        });
    }
}

fn min_max_u32(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("min_u32");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let data: Vec<_> = (0..size).map(|_| rng.next_u32()).collect();

        g.bench_with_input(BenchmarkId::new("simd", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data.min_simd());
            });
        });

        g.bench_with_input(BenchmarkId::new("iter", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(min_iter(&data));
            });
        });
    }

    g.finish();

    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("max_u32");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let data: Vec<_> = (0..size).map(|_| rng.next_u32()).collect();

        g.bench_with_input(BenchmarkId::new("simd", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data.max_simd());
            });
        });

        g.bench_with_input(BenchmarkId::new("iter", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(max_iter(&data));
            });
        });
    }
}

fn min_max_f32(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("min_f32");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let data: Vec<_> = (0..size).map(|_| rng.next_u32() as f32).collect();

        g.bench_with_input(BenchmarkId::new("simd", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(black_box(&data).min_simd());
            });
        });

        g.bench_with_input(BenchmarkId::new("iter", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(min_iter_f32(black_box(&data)));
            });
        });
    }

    g.finish();

    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("max_f32");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let data: Vec<_> = (0..size).map(|_| rng.next_u32() as f32).collect();

        g.bench_with_input(BenchmarkId::new("simd", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(black_box(&data).max_simd());
            });
        });

        g.bench_with_input(BenchmarkId::new("iter", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(max_iter_f32(black_box(&data)));
            });
        });
    }
}

fn contains_u32(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("contains_u32");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let needle = rng.next_u32();
        let data: Vec<_> = (0..size)
            .map(|_| {
                loop {
                    let r = rng.next_u32();
                    if r != needle {
                        break r;
                    }
                }
            })
            .collect();
        assert!(!data.contains(&needle));

        let mut data_avg = data.clone();
        data_avg[data.len() / 2] = needle;

        let mut data_worst = data.clone();
        data_worst[size - 1] = needle;

        g.bench_with_input(BenchmarkId::new("simd-avg", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data_avg.contains_simd(needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("iter-avg", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(contains_iter(&data_avg, needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("simd-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data_worst.contains_simd(needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("iter-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(contains_iter(&data_worst, needle));
            });
        });
    }
}

#[inline]
fn find_iter(array: &[u32], needle: u32) -> Option<usize> {
    array.iter().position(|i| *i == needle)
}

#[inline]
fn contains_iter(array: &[u32], needle: u32) -> bool {
    array.iter().any(|i| *i == needle)
}

#[inline]
fn count_iter(array: &[u32], needle: u32) -> usize {
    array.iter().filter(|i| **i == needle).count()
}

#[inline]
fn min_iter(array: &[u32]) -> Option<u32> {
    array.iter().min().copied()
}

#[inline]
fn min_iter_f32(array: &[f32]) -> Option<f32> {
    if array.is_empty() {
        return None;
    }

    let mut min = array[0];

    for i in &array[1..] {
        if *i < min {
            min = *i;
        }
    }

    Some(min)
}

#[inline]
fn max_iter_f32(array: &[f32]) -> Option<f32> {
    if array.is_empty() {
        return None;
    }

    let mut max = array[0];

    for i in &array[1..] {
        if *i > max {
            max = *i;
        }
    }

    Some(max)
}

#[inline]
fn max_iter(array: &[u32]) -> Option<u32> {
    array.iter().max().copied()
}

criterion_group!(
    benches,
    find_u32,
    count_u32,
    min_max_u32,
    min_max_f32,
    contains_u32
);
criterion_main!(benches);
