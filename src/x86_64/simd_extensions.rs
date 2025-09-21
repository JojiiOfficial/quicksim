use std::{
    arch::x86_64::{
        __m256, __m256i, _mm_add_epi32, _mm_extract_epi32, _mm_extract_ps, _mm_hadd_epi32,
        _mm_max_epu8, _mm_max_epu32, _mm_max_ps, _mm_min_epu8, _mm_min_epu32, _mm_min_ps,
        _mm_shuffle_epi32, _mm_shuffle_ps, _mm256_castps256_ps128, _mm256_castsi256_si128,
        _mm256_extractf128_ps, _mm256_extracti128_si256,
    },
    mem::transmute,
};

/// Calculates the horizontal sum of 8x 32bit integers.
#[target_feature(enable = "avx2")]
pub fn negative_horizontal_sum_u32_avx(input: __m256i) -> u32 {
    let sum_128 = _mm_add_epi32(
        _mm256_castsi256_si128(input),
        _mm256_extracti128_si256::<1>(input),
    );
    let hsum = _mm_hadd_epi32(sum_128, sum_128);
    (-(_mm_extract_epi32::<0>(hsum) as i32) + (-(_mm_extract_epi32::<1>(hsum) as i32))) as u32
}

/// Calculates the horizontal sum of 8x 32bit integers.
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub fn horizontal_sum_u32_avx(input: __m256i) -> u32 {
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(input),
        _mm256_extracti128_si256::<1>(input),
    );
    let hsum = _mm_hadd_epi32(sum128, sum128);
    _mm_extract_epi32::<0>(hsum) as u32 + _mm_extract_epi32::<1>(hsum) as u32
}

/// Calculates the horizontal maximum of 32x u8.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn horizontal_max_u8_avx(a: __m256i) -> u8 {
    let min128 = _mm_max_epu8(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));

    // Safety: we can safely transmute a __m128i to [u8; 16]
    let array: [u8; 16] = unsafe { transmute(min128) };

    // Safety: `array` is always of length 16.
    unsafe { *array.iter().max().unwrap_unchecked() }
}

/// Calculates the horizontal minimum of 32x u8.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn horizontal_min_u8_avx(a: __m256i) -> u8 {
    let min128 = _mm_min_epu8(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));

    // Safety: we can safely transmute a __m128i to [u8; 16]
    let array: [u8; 16] = unsafe { transmute(min128) };

    // Safety: `array` is always of length 16.
    unsafe { *array.iter().min().unwrap_unchecked() }
}

/// Calculates the horizontal minimum of 8x u32.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn horizontal_min_u32_avx(a: __m256i) -> u32 {
    let min128 = _mm_min_epu32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
    let min128_shuffled = _mm_shuffle_epi32(min128, 0b01_00_11_10);
    let min64 = _mm_min_epu32(min128, min128_shuffled);
    (_mm_extract_epi32::<0>(min64) as u32).min(_mm_extract_epi32::<1>(min64) as u32)
}

/// Calculates the horizontal maximum of 8x u32.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn horizontal_max_u32_avx(a: __m256i) -> u32 {
    let min128 = _mm_max_epu32(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
    let min128_shuffled = _mm_shuffle_epi32(min128, 0b01_00_11_10);
    let min64 = _mm_max_epu32(min128, min128_shuffled);
    (_mm_extract_epi32::<0>(min64) as u32).max(_mm_extract_epi32::<1>(min64) as u32)
}

/// Calculates the horizontal minimum of 8x f32.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn horizontal_min_f32_avx(a: __m256) -> f32 {
    let min128 = _mm_min_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
    let min128_shuffled = _mm_shuffle_ps(min128, min128, 0b01_00_11_10);
    let min64 = _mm_min_ps(min128, min128_shuffled);

    let hi = f32::from_bits(_mm_extract_ps::<0>(min64) as u32);
    let lo = f32::from_bits(_mm_extract_ps::<1>(min64) as u32);

    if hi < lo { hi } else { lo }
}

/// Calculates the horizontal minimum of 8x f32.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn horizontal_max_f32_avx(a: __m256) -> f32 {
    let min128 = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a));
    let min128_shuffled = _mm_shuffle_ps(min128, min128, 0b01_00_11_10);
    let min64 = _mm_max_ps(min128, min128_shuffled);

    let hi = f32::from_bits(_mm_extract_ps::<0>(min64) as u32);
    let lo = f32::from_bits(_mm_extract_ps::<1>(min64) as u32);
    if hi > lo { hi } else { lo }
}

#[cfg(test)]
mod test {
    use std::arch::x86_64::{_mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256};

    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use super::*;

    #[test]
    fn test_horizontal_sum() {
        unsafe {
            // Test all 1
            let input = _mm256_set1_epi32(1);
            let hsum = horizontal_sum_u32_avx(input);
            assert_eq!(hsum, 8);

            // Test arbitrary values
            let input = [1, 10, 2, 15, 3, 20, 4, 25];
            let input_reg = _mm256_setr_epi32(
                input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
            );
            let hsum = horizontal_sum_u32_avx(input_reg);
            let expected = input.iter().sum::<i32>() as u32;
            assert_eq!(hsum, expected);

            // Test all 0
            let input = _mm256_setzero_si256();
            assert_eq!(horizontal_sum_u32_avx(input), 0);
        }
    }

    #[test]
    fn test_horizontal_min_max() {
        let mut rng = StdRng::seed_from_u64(42);

        unsafe {
            for _ in 0..1000 {
                let input: [i32; 8] = (0..8)
                    .map(|_| rng.next_u32() as i32)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let input_reg = _mm256_setr_epi32(
                    input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
                );

                let min = horizontal_min_u32_avx(input_reg);
                let real_min = input.iter().map(|i| *i as u32).min().unwrap();
                assert_eq!(min, real_min);

                let max = horizontal_max_u32_avx(input_reg);
                let real_max = input.iter().map(|i| *i as u32).max().unwrap();
                assert_eq!(max, real_max);
            }
        }
    }
}

// #[target_feature(enable = "avx")]
// fn _mm256_debug_epi32(inp: __m256i) {
//     let t: [u32; 8] = unsafe { transmute(inp) };
//     println!("{t:?}");
// }

// #[target_feature(enable = "avx")]
// fn _mm_debug_epi32(pre: &str, inp: __m128i) {
//     let t: [u32; 4] = unsafe { transmute(inp) };
//     println!("{t:?}\t{pre}");
// }
