use std::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_max_epu32,
    _mm256_min_epu32, _mm256_movemask_epi8, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_testz_si256,
};

use crate::x86_64::simd_extensions::{
    horizontal_max_u32_avx, horizontal_min_u32_avx, negative_horizontal_sum_u32_avx,
};

/// Returns `true` if `needle` is an elemen in the given array.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn contains_avx(array: &[u32], needle: u32) -> bool {
    const STEP: usize = 32;

    let needle_mask = _mm256_set1_epi32(needle as i32);

    let len = array.len();
    let m = len % STEP;
    let vectorized_part = len - m;
    let mut i = 0;

    let mut ptr = array.as_ptr();

    unsafe {
        while i < vectorized_part {
            let curr_items = _mm256_loadu_si256(ptr.cast::<__m256i>());
            let curr_items_p1 = _mm256_loadu_si256(ptr.add(8).cast::<__m256i>());
            let curr_items_p2 = _mm256_loadu_si256(ptr.add(16).cast::<__m256i>());
            let curr_items_p3 = _mm256_loadu_si256(ptr.add(24).cast::<__m256i>());

            let compared = _mm256_cmpeq_epi32(needle_mask, curr_items);
            let compared1 = _mm256_cmpeq_epi32(needle_mask, curr_items_p1);
            let compared2 = _mm256_cmpeq_epi32(needle_mask, curr_items_p2);
            let compared3 = _mm256_cmpeq_epi32(needle_mask, curr_items_p3);

            if _mm256_testz_si256(compared, compared) == 0 {
                return true;
            }

            if _mm256_testz_si256(compared1, compared1) == 0 {
                return true;
            }

            if _mm256_testz_si256(compared2, compared2) == 0 {
                return true;
            }

            if _mm256_testz_si256(compared3, compared3) == 0 {
                return true;
            }

            ptr = ptr.add(STEP);
            i += STEP;
        }
    }

    array[vectorized_part..].iter().any(|i| *i == needle)
}

/// Returns the position of `needle` in `array` if the array contains it.
/// If there are multiple occurrences of `needle` in `array`, the first index gets returned.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn find_avx(array: &[u32], needle: u32) -> Option<usize> {
    const STEP: usize = 32;

    let needle_mask = _mm256_set1_epi32(needle as i32);

    let len = array.len();
    let m = len % STEP;
    let vectorized_part = len - m;
    let mut i = 0;

    let mut ptr = array.as_ptr();

    unsafe {
        while i < vectorized_part {
            let curr_items = _mm256_loadu_si256(ptr.cast::<__m256i>());
            let curr_items_p1 = _mm256_loadu_si256(ptr.add(8).cast::<__m256i>());
            let curr_items_p2 = _mm256_loadu_si256(ptr.add(16).cast::<__m256i>());
            let curr_items_p3 = _mm256_loadu_si256(ptr.add(24).cast::<__m256i>());

            let compared = _mm256_cmpeq_epi32(needle_mask, curr_items);
            let compared1 = _mm256_cmpeq_epi32(needle_mask, curr_items_p1);
            let compared2 = _mm256_cmpeq_epi32(needle_mask, curr_items_p2);
            let compared3 = _mm256_cmpeq_epi32(needle_mask, curr_items_p3);

            if _mm256_testz_si256(compared, compared) == 0 {
                let mask = _mm256_movemask_epi8(compared);
                let res = mask.trailing_zeros() / 4;
                return Some(res as usize + i);
            }

            if _mm256_testz_si256(compared1, compared1) == 0 {
                let mask = _mm256_movemask_epi8(compared1);
                let res = mask.trailing_zeros() / 4;
                return Some(res as usize + i + 8);
            }

            if _mm256_testz_si256(compared2, compared2) == 0 {
                let mask = _mm256_movemask_epi8(compared2);
                let res = mask.trailing_zeros() / 4;
                return Some(res as usize + i + 16);
            }

            if _mm256_testz_si256(compared3, compared3) == 0 {
                let mask = _mm256_movemask_epi8(compared3);
                let res = mask.trailing_zeros() / 4;
                return Some(res as usize + i + 24);
            }

            ptr = ptr.add(STEP);
            i += STEP;
        }
    }

    array[vectorized_part..]
        .iter()
        .position(|i| *i == needle)
        .map(|remainder_pos| remainder_pos + vectorized_part)
}

/// Returns the amount of occurrences of `needle` in `array`.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn count_avx(array: &[u32], element: u32) -> usize {
    const STEP: usize = 32;

    let needle_mask = _mm256_set1_epi32(element as i32);
    let len = array.len();
    let m = len % STEP;
    let iterr = len - m;
    let mut i = 0;

    let mut ptr = array.as_ptr();

    let mut sum = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let mut sum4 = _mm256_setzero_si256();

    unsafe {
        while i < iterr {
            let curr_items = _mm256_loadu_si256(ptr.cast::<__m256i>());
            let curr_items_2 = _mm256_loadu_si256(ptr.add(8).cast::<__m256i>());
            let curr_items_3 = _mm256_loadu_si256(ptr.add(16).cast::<__m256i>());
            let curr_items_4 = _mm256_loadu_si256(ptr.add(24).cast::<__m256i>());

            let cmp = _mm256_cmpeq_epi32(curr_items, needle_mask);
            let cmp2 = _mm256_cmpeq_epi32(curr_items_2, needle_mask);
            let cmp3 = _mm256_cmpeq_epi32(curr_items_3, needle_mask);
            let cmp4 = _mm256_cmpeq_epi32(curr_items_4, needle_mask);

            sum = _mm256_add_epi32(sum, cmp);
            sum2 = _mm256_add_epi32(sum2, cmp2);
            sum3 = _mm256_add_epi32(sum3, cmp3);
            sum4 = _mm256_add_epi32(sum4, cmp4);

            i += STEP;
            ptr = ptr.add(STEP);
        }
    }

    let t1 = _mm256_add_epi32(sum, sum2);
    let t2 = _mm256_add_epi32(sum3, sum4);
    let sum = _mm256_add_epi32(t1, t2);
    let simd_res = negative_horizontal_sum_u32_avx(sum) as usize;

    let remainder = array[iterr..].iter().filter(|i| **i == element).count();
    remainder + simd_res
}

/// Returns the smallest item in the array, or `None` if the array was empty.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn min_avx(array: &[u32]) -> Option<u32> {
    const STEP: usize = 32;

    let len = array.len();
    let m = len % STEP;
    let iterr = len - m;

    let mut i = 0;

    let mut ptr = array.as_ptr();

    let mut lmins1 = _mm256_set1_epi32(u32::MAX as i32);
    let mut lmins2 = _mm256_set1_epi32(u32::MAX as i32);
    let mut lmins3 = _mm256_set1_epi32(u32::MAX as i32);
    let mut lmins4 = _mm256_set1_epi32(u32::MAX as i32);

    unsafe {
        while i < iterr {
            let current = _mm256_loadu_si256(ptr.cast());
            let current2 = _mm256_loadu_si256(ptr.add(8).cast());
            let current3 = _mm256_loadu_si256(ptr.add(16).cast());
            let current4 = _mm256_loadu_si256(ptr.add(24).cast());

            lmins1 = _mm256_min_epu32(current, lmins1);
            lmins2 = _mm256_min_epu32(current2, lmins2);
            lmins3 = _mm256_min_epu32(current3, lmins3);
            lmins4 = _mm256_min_epu32(current4, lmins4);

            i += STEP;
            ptr = ptr.add(STEP);
        }
    }

    let m1 = _mm256_min_epu32(lmins1, lmins2);
    let m2 = _mm256_min_epu32(lmins3, lmins4);
    let lmins = _mm256_min_epu32(m1, m2);
    let min = horizontal_min_u32_avx(lmins);

    if let Some(remainer_min) = array[iterr..].iter().min() {
        return Some(min.min(*remainer_min));
    }

    Some(min)
}

/// Returns the largest item in the array, or `None` if the array was empty.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn max_avx(array: &[u32]) -> Option<u32> {
    const STEP: usize = 32;

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
            let current2 = _mm256_loadu_si256(ptr.add(8).cast());
            let current3 = _mm256_loadu_si256(ptr.add(16).cast());
            let current4 = _mm256_loadu_si256(ptr.add(24).cast());

            lmax1 = _mm256_max_epu32(current, lmax1);
            lmax2 = _mm256_max_epu32(current2, lmax2);
            lmax3 = _mm256_max_epu32(current3, lmax3);
            lmax4 = _mm256_max_epu32(current4, lmax4);

            i += STEP;
            ptr = ptr.add(STEP);
        }
    }

    let m1 = _mm256_max_epu32(lmax1, lmax2);
    let m2 = _mm256_max_epu32(lmax3, lmax4);
    let lmins = _mm256_max_epu32(m1, m2);
    let max = horizontal_max_u32_avx(lmins);

    if let Some(remainer_max) = array[iterr..].iter().max() {
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
        needle: u32,
    ) -> Vec<u32> {
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
        value: Option<u32>,
        index: Option<usize>,
    ) -> Vec<u32> {
        let mut vec: Vec<_> = (0..len)
            .map(|_| {
                let mut r = rng.next_u32();

                // If a custom was provided, ensure it's not part of the base array.
                if let Some(val) = value
                    && r == val
                {
                    loop {
                        r = rng.next_u32();
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

        let value = rng.next_u32();
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
            let value = rng.next_u32();
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

    #[rstest]
    // (len, count)
    #[case(32, 0)]
    #[case(32, 1)]
    #[case(32, 10)]
    #[case(33, 0)]
    #[case(33, 5)]
    #[case(33, 20)]
    #[case(64, 0)]
    #[case(64, 10)]
    #[case(64, 60)]
    #[case(65, 20)]
    #[case(128, 0)]
    #[case(128, 2)]
    #[case(128, 4)]
    fn test_array_count(#[case] len: usize, #[case] count: usize) {
        let mut rng = StdRng::seed_from_u64(42);
        let needle = rng.next_u32();

        let vec = random_array_with_count(&mut rng, len, count, needle);
        let simd_count = unsafe { count_avx(&vec, needle) };
        assert_eq!(simd_count, count);
    }

    #[test]
    fn test_array_count_fuzzy() {
        let mut rng = StdRng::seed_from_u64(42);
        let needle = rng.next_u32();

        for len in [32, 64, 127, 128, 256, 513] {
            for count in 0..len {
                let vec = random_array_with_count(&mut rng, len, count, needle);
                let simd_count = unsafe { count_avx(&vec, needle) };
                assert_eq!(simd_count, count);
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
