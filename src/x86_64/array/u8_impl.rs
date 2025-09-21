use std::arch::x86_64::*;

use crate::x86_64::simd_extensions::{horizontal_max_u8_avx, horizontal_min_u8_avx};

/// Returns `true` if `needle` is an elemen in the given array.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn contains_avx(array: &[u8], needle: u8) -> bool {
    const STEP: usize = 128;

    let needle_mask = _mm256_set1_epi8(needle as i8);

    let len = array.len();
    let m = len % STEP;
    let vectorized_part = len - m;
    let mut i = 0;

    let mut ptr = array.as_ptr();

    unsafe {
        while i < vectorized_part {
            let curr_items = _mm256_loadu_si256(ptr.cast());
            let curr_items_p1 = _mm256_loadu_si256(ptr.add(32).cast());
            let curr_items_p2 = _mm256_loadu_si256(ptr.add(64).cast());
            let curr_items_p3 = _mm256_loadu_si256(ptr.add(96).cast());

            let compared = _mm256_cmpeq_epi8(needle_mask, curr_items);
            let compared1 = _mm256_cmpeq_epi8(needle_mask, curr_items_p1);
            let compared2 = _mm256_cmpeq_epi8(needle_mask, curr_items_p2);
            let compared3 = _mm256_cmpeq_epi8(needle_mask, curr_items_p3);

            if _mm256_testz_si256(compared, compared) == 0
                || _mm256_testz_si256(compared1, compared1) == 0
                || _mm256_testz_si256(compared2, compared2) == 0
                || _mm256_testz_si256(compared3, compared3) == 0
            {
                return true;
            }

            ptr = ptr.add(STEP);
            i += STEP;
        }
    }

    if m == 0 {
        return false;
    }

    const STEP_HALF: usize = 32;

    let half_m = m % STEP_HALF;
    let remain_vectorized_part = m - half_m;
    i = 0;

    let needle_mask = _mm256_castsi256_si128(needle_mask);

    unsafe {
        while i < remain_vectorized_part {
            let curr_items = _mm_loadu_si128(ptr.cast());
            let curr_items_p1 = _mm_loadu_si128(ptr.add(16).cast());

            let compared = _mm_cmpeq_epi8(needle_mask, curr_items);
            let compared1 = _mm_cmpeq_epi8(needle_mask, curr_items_p1);

            if _mm_testz_si128(compared, compared) == 0
                || _mm_testz_si128(compared1, compared1) == 0
            {
                return true;
            }

            ptr = ptr.add(STEP_HALF);
            i += STEP_HALF;
        }
    }

    array[(len - half_m)..].iter().any(|i| *i == needle)
}
/// Returns `true` if `needle` is an elemen in the given array.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn find_avx(array: &[u8], needle: u8) -> Option<usize> {
    const STEP: usize = 128;

    let needle_mask = _mm256_set1_epi8(needle as i8);

    let len = array.len();
    let m = len % STEP;
    let vectorized_part = len - m;
    let mut i = 0;

    let mut ptr = array.as_ptr();

    unsafe {
        while i < vectorized_part {
            let curr_items = _mm256_loadu_si256(ptr.cast());
            let curr_items_p1 = _mm256_loadu_si256(ptr.add(32).cast());
            let curr_items_p2 = _mm256_loadu_si256(ptr.add(64).cast());
            let curr_items_p3 = _mm256_loadu_si256(ptr.add(96).cast());

            let compared = _mm256_cmpeq_epi8(needle_mask, curr_items);
            let compared1 = _mm256_cmpeq_epi8(needle_mask, curr_items_p1);
            let compared2 = _mm256_cmpeq_epi8(needle_mask, curr_items_p2);
            let compared3 = _mm256_cmpeq_epi8(needle_mask, curr_items_p3);

            if _mm256_testz_si256(compared, compared) == 0 {
                let mask = _mm256_movemask_epi8(compared);
                let res = mask.trailing_zeros();
                return Some(res as usize + i);
            }

            if _mm256_testz_si256(compared1, compared1) == 0 {
                let mask = _mm256_movemask_epi8(compared1);
                let res = mask.trailing_zeros();
                return Some(res as usize + i + 32);
            }

            if _mm256_testz_si256(compared2, compared2) == 0 {
                let mask = _mm256_movemask_epi8(compared2);
                let res = mask.trailing_zeros();
                return Some(res as usize + i + 64);
            }

            if _mm256_testz_si256(compared3, compared3) == 0 {
                let mask = _mm256_movemask_epi8(compared3);
                let res = mask.trailing_zeros();
                return Some(res as usize + i + 96);
            }

            ptr = ptr.add(STEP);
            i += STEP;
        }
    }

    if m == 0 {
        return None;
    }

    const STEP_HALF: usize = 32;

    let half_m = m % STEP_HALF;
    let remain_vectorized_part = m - half_m;
    i = 0;

    let needle_mask = _mm256_castsi256_si128(needle_mask);

    unsafe {
        while i < remain_vectorized_part {
            let curr_items = _mm_loadu_si128(ptr.cast());
            let curr_items_p1 = _mm_loadu_si128(ptr.add(16).cast());

            let compared = _mm_cmpeq_epi8(needle_mask, curr_items);
            let compared1 = _mm_cmpeq_epi8(needle_mask, curr_items_p1);

            if _mm_testz_si128(compared, compared) == 0 {
                let mask = _mm_movemask_epi8(compared);
                let res = mask.trailing_zeros();
                return Some(res as usize + i);
            }

            if _mm_testz_si128(compared1, compared1) == 0 {
                let mask = _mm_movemask_epi8(compared1);
                let res = mask.trailing_zeros();
                return Some(res as usize + i + 16);
            }

            ptr = ptr.add(STEP_HALF);
            i += STEP_HALF;
        }
    }

    let remaining = len - half_m;
    array[remaining..]
        .iter()
        .position(|i| *i == needle)
        .map(|i| remaining + i)
}

/// Returns the smallest item in the array, or `None` if the array was empty.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn min_avx(array: &[u8]) -> Option<u8> {
    const STEP: usize = 128;

    let len = array.len();
    let m = len % STEP;
    let iterr = len - m;

    let mut i = 0;

    let mut ptr = array.as_ptr();

    let mut lmins1 = _mm256_set1_epi8(u8::MAX as i8);
    let mut lmins2 = _mm256_set1_epi8(u8::MAX as i8);
    let mut lmins3 = _mm256_set1_epi8(u8::MAX as i8);
    let mut lmins4 = _mm256_set1_epi8(u8::MAX as i8);

    unsafe {
        while i < iterr {
            let current = _mm256_loadu_si256(ptr.cast());
            let current2 = _mm256_loadu_si256(ptr.add(32).cast());
            let current3 = _mm256_loadu_si256(ptr.add(64).cast());
            let current4 = _mm256_loadu_si256(ptr.add(96).cast());

            lmins1 = _mm256_min_epu8(current, lmins1);
            lmins2 = _mm256_min_epu8(current2, lmins2);
            lmins3 = _mm256_min_epu8(current3, lmins3);
            lmins4 = _mm256_min_epu8(current4, lmins4);

            i += STEP;
            ptr = ptr.add(STEP);
        }
    }

    let m1 = _mm256_min_epu8(lmins1, lmins2);
    let m2 = _mm256_min_epu8(lmins3, lmins4);
    let mins = _mm256_min_epu8(m1, m2);

    if m == 0 {
        return Some(horizontal_min_u8_avx(mins));
    }

    // Split 256 lane to 2x 128 so we can continue with half-width.
    let mut lmins1 = _mm256_castsi256_si128(mins);
    let mut lmins2 = _mm256_extracti128_si256::<1>(mins);

    const STEP_HALF: usize = 32;

    let half_m = m % STEP_HALF;
    let remain_vectorized_part = m - half_m;
    i = 0;

    unsafe {
        while i < remain_vectorized_part {
            let current = _mm_loadu_si128(ptr.cast());
            let current2 = _mm_loadu_si128(ptr.add(16).cast());

            lmins1 = _mm_min_epu8(current, lmins1);
            lmins2 = _mm_min_epu8(current2, lmins2);

            i += STEP_HALF;
            ptr = ptr.add(STEP_HALF)
        }
    }

    let mins = _mm_min_epu8(lmins1, lmins2);
    // Safety: we can safely transmute a __m128i to [u8; 16]
    let mins_array: [u8; 16] = unsafe { std::mem::transmute(mins) };
    // Safety: `array` is always of length 16.
    let min = unsafe { *mins_array.iter().min().unwrap_unchecked() };

    let remaining = len - half_m;

    if let Some(remainer_min) = array[remaining..].iter().min() {
        return Some(min.min(*remainer_min));
    }

    Some(min)
}

/// Returns the smallest item in the array, or `None` if the array was empty.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn max_avx(array: &[u8]) -> Option<u8> {
    const STEP: usize = 128;

    let len = array.len();
    let m = len % STEP;
    let iterr = len - m;

    let mut i = 0;

    let mut ptr = array.as_ptr();

    let mut lmax1 = _mm256_setzero_si256();
    let mut lmax2 = _mm256_setzero_si256();
    let mut lmax3 = _mm256_setzero_si256();
    let mut lmax4 = _mm256_setzero_si256();

    unsafe {
        while i < iterr {
            let current = _mm256_loadu_si256(ptr.cast());
            let current2 = _mm256_loadu_si256(ptr.add(32).cast());
            let current3 = _mm256_loadu_si256(ptr.add(64).cast());
            let current4 = _mm256_loadu_si256(ptr.add(96).cast());

            lmax1 = _mm256_max_epu8(current, lmax1);
            lmax2 = _mm256_max_epu8(current2, lmax2);
            lmax3 = _mm256_max_epu8(current3, lmax3);
            lmax4 = _mm256_max_epu8(current4, lmax4);

            i += STEP;
            ptr = ptr.add(STEP);
        }
    }

    let m1 = _mm256_max_epu8(lmax1, lmax2);
    let m2 = _mm256_max_epu8(lmax3, lmax4);
    let mins = _mm256_max_epu8(m1, m2);

    if m == 0 {
        return Some(horizontal_max_u8_avx(mins));
    }

    // Split 256 lane to 2x 128 so we can continue with half-width.
    let mut lmax1 = _mm256_castsi256_si128(mins);
    let mut lmax2 = _mm256_extracti128_si256::<1>(mins);

    const STEP_HALF: usize = 32;

    let half_m = m % STEP_HALF;
    let remain_vectorized_part = m - half_m;
    i = 0;

    unsafe {
        while i < remain_vectorized_part {
            let current = _mm_loadu_si128(ptr.cast());
            let current2 = _mm_loadu_si128(ptr.add(16).cast());

            lmax1 = _mm_max_epu8(current, lmax1);
            lmax2 = _mm_max_epu8(current2, lmax2);

            i += STEP_HALF;
            ptr = ptr.add(STEP_HALF)
        }
    }

    let mins = _mm_max_epu8(lmax1, lmax2);
    // Safety: we can safely transmute a __m128i to [u8; 16]
    let mins_array: [u8; 16] = unsafe { std::mem::transmute(mins) };
    // Safety: `array` is always of length 16.
    let max = unsafe { *mins_array.iter().max().unwrap_unchecked() };

    let remaining = len - half_m;

    if let Some(remainer_max) = array[remaining..].iter().max() {
        return Some(max.max(*remainer_max));
    }

    Some(max)
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use rand::rngs::StdRng;
    use rand::{Rng, RngCore, SeedableRng};
    use rstest::rstest;

    use super::*;

    fn random_array_with_count(
        rng: &mut impl Rng,
        len: usize,
        count: usize,
        needle: u8,
    ) -> Vec<u8> {
        assert!(count <= len);

        let mut vec = random_array_with_value(rng, len, Some(needle), None);

        let mut indices: HashSet<usize> = HashSet::new();

        for _ in 0..count {
            loop {
                let next_index = (rng.next_u64() % len as u64) as usize;

                if !indices.contains(&next_index) {
                    indices.insert(next_index);
                    break;
                }
            }
        }

        assert_eq!(indices.len(), count);

        for index in indices {
            vec[index] = needle;
        }

        vec
    }

    pub fn random_array_with_value(
        rng: &mut impl Rng,
        len: usize,
        value: Option<u8>,
        index: Option<usize>,
    ) -> Vec<u8> {
        let mut vec: Vec<_> = (0..len)
            .map(|_| {
                let mut r = (rng.next_u32() % u8::MAX as u32) as u8;

                // If a custom was provided, ensure it's not part of the base array.
                if let Some(val) = value
                    && r == val
                {
                    loop {
                        r = (rng.next_u32() % u8::MAX as u32) as u8;
                        if r != val {
                            break;
                        }
                    }
                }

                r
            })
            .collect();

        match (value, index) {
            (Some(val), Some(index)) => {
                if index < len {
                    vec[index] = val;
                }
            }
            _ => (),
        }

        vec
    }

    #[test]
    fn test_contains_u8_fuzzy() {
        let mut rng = StdRng::seed_from_u64(42);

        for size in [16, 32, 64, 128, 256, 300, 512, 513, 1030, 8000] {
            for _ in 0..100 {
                let needle = (rng.next_u32() % u8::MAX as u32) as u8;
                let has = rng.random_bool(0.9);
                let count = if has { 1 } else { 0 };
                let vec = random_array_with_count(&mut rng, size, count, needle);

                let simd_contains = unsafe { contains_avx(&vec, needle) };
                if simd_contains != has {
                    println!("{:?} {} {vec:?}", size, needle)
                }
                assert_eq!(simd_contains, has);
            }
        }
    }

    #[rstest]
    // (len, value, index)
    #[case::test_empty_simd(63, None)]
    #[case::test_empty_simd(64, None)]
    #[case::test_empty_simd(65, None)]
    #[case::test_empty_simd(66, None)]
    #[case::test_non_existend_no_simd(5, None)]
    #[case::test_non_existend_simd(128, None)]
    #[case::test_simd_first(64, Some(0))]
    #[case::test_simd_arbitrary(64, Some(12))]
    #[case::test_simd_last(64, Some(63))]
    #[case::test_simd_first(65, Some(0))]
    #[case::test_simd_arbitrary(65, Some(12))]
    #[case::test_simd_arbitrary(65, Some(31))]
    #[case::test_simd_arbitrary(65, Some(32))]
    #[case::test_simd_arbitrary(65, Some(33))]
    #[case::test_simd_arbitrary(33, Some(30))]
    #[case::test_simd_arbitrary(32, Some(30))]
    #[case::test_simd_arbitrary(32, Some(31))]
    #[case::test_simd_arbitrary(65, Some(15))]
    #[case::test_simd_arbitrary(65, Some(16))]
    #[case::test_simd_arbitrary(65, Some(17))]
    #[case::test_simd_last(65, Some(64))]
    #[case::test_simd_arbitrary(65, Some(12))]
    #[case::test_simd_first(128, Some(0))]
    #[case::test_simd_arbitrary(128, Some(64))]
    #[case::test_simd_arbitrary(128, Some(65))]
    #[case::test_simd_arbitrary(128, Some(66))]
    #[case::test_simd_arbitrary(128, Some(90))]
    #[case::test_simd_arbitrary(128, Some(102))]
    #[case::test_simd_arbitrary(128, Some(126))]
    #[case::test_simd_arbitrary(128, Some(127))]
    fn test_array_find(#[case] len: usize, #[case] index: Option<usize>) {
        let mut rng = StdRng::seed_from_u64(42);

        let value = (rng.next_u32() % u8::MAX as u32) as u8;
        let vec = random_array_with_value(&mut rng, len, Some(value), index);

        let simd_result = unsafe { find_avx(&vec, value) };
        assert_eq!(simd_result, index, "{value} in {vec:?}");

        let contains = unsafe { contains_avx(&vec, value) };
        assert_eq!(contains, simd_result.is_some());
    }

    #[test]
    fn test_array_find_fuzzy() {
        let mut rng = StdRng::seed_from_u64(42);

        for size in [32, 64, 127, 128, 256, 513] {
            let value = (rng.next_u32() % u8::MAX as u32) as u8;
            for index in 0..size {
                for has in [true, false] {
                    let index = has.then_some(index);

                    let vec = random_array_with_value(&mut rng, size, Some(value), index);
                    let simd_result = unsafe { find_avx(&vec, value) };
                    assert_eq!(simd_result, index, "{value} in {vec:?}");

                    let contains = unsafe { contains_avx(&vec, value) };
                    assert_eq!(contains, has);
                }
            }
        }
    }

    #[test]
    fn test_array_min_max_fuzzy() {
        let mut rng = StdRng::seed_from_u64(42);

        for len in [32, 64, 127, 128, 256, 513, 1024, 6256] {
            let vec = random_array_with_value(&mut rng, len, None, None);

            let simd_min = unsafe { min_avx(&vec) };
            let real_min = vec.iter().min().copied();
            assert_eq!(simd_min, real_min);

            let simd_max = unsafe { max_avx(&vec) };
            let real_max = vec.iter().max().copied();
            assert_eq!(simd_max, real_max);
        }
    }
}
