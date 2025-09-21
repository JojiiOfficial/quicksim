use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use quicksim::traits::array_u8::ArrayU8SimdExt;
use rand::{RngCore, SeedableRng, rngs::StdRng};

const U32_ARRAY_LEN_TO_CHECK: [usize; 9] = [32, 64, 128, 130, 168, 530, 1028, 5010, 8000];

fn contains_u8(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("contains_u8");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let needle = (rng.next_u32() % u8::MAX as u32) as u8;
        let data: Vec<u8> = (0..size)
            .map(|_| {
                loop {
                    let r = (rng.next_u32() % u8::MAX as u32) as u8;
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
                let _ = black_box(contains_iter_u8(&data_avg, needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("simd-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data_worst.contains_simd(needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("iter-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(contains_iter_u8(&data_worst, needle));
            });
        });
    }
}

fn find_u8(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("find_u8");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let needle = (rng.next_u32() % u8::MAX as u32) as u8;
        let data: Vec<u8> = (0..size)
            .map(|_| {
                loop {
                    let r = (rng.next_u32() % u8::MAX as u32) as u8;
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
                let _ = black_box(find_iter_u8(&data_avg, needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("simd-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(data_worst.find_simd(needle));
            });
        });

        g.bench_with_input(BenchmarkId::new("iter-worst", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(find_iter_u8(&data_worst, needle));
            });
        });
    }
}

fn min_max_u8(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("min_u8");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let data: Vec<u8> = (0..size)
            .map(|_| (rng.next_u32() % u8::MAX as u32) as u8)
            .collect();

        g.bench_with_input(BenchmarkId::new("simd", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(black_box(&data).min_simd());
            });
        });

        g.bench_with_input(BenchmarkId::new("iter", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(min_iter_u8(black_box(&data)));
            });
        });
    }

    g.finish();

    let mut rng = StdRng::seed_from_u64(42);

    let mut g = c.benchmark_group("max_u8");

    for size in U32_ARRAY_LEN_TO_CHECK {
        let data: Vec<u8> = (0..size)
            .map(|_| (rng.next_u32() % u8::MAX as u32) as u8)
            .collect();

        g.bench_with_input(BenchmarkId::new("simd", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(black_box(&data).max_simd());
            });
        });

        g.bench_with_input(BenchmarkId::new("iter", size), &size, |i, _| {
            i.iter(|| {
                let _ = black_box(max_iter_u8(black_box(&data)));
            });
        });
    }
}

#[inline]
pub(crate) fn contains_iter_u8(array: &[u8], needle: u8) -> bool {
    array.iter().any(|i| *i == needle)
}

#[inline]
pub(crate) fn find_iter_u8(array: &[u8], needle: u8) -> Option<usize> {
    array.iter().position(|i| *i == needle)
}

#[inline]
fn min_iter_u8(array: &[u8]) -> Option<u8> {
    array.iter().min().copied()
}

#[inline]
fn max_iter_u8(array: &[u8]) -> Option<u8> {
    array.iter().max().copied()
}

criterion_group!(benches, contains_u8, find_u8, min_max_u8);
criterion_main!(benches);
