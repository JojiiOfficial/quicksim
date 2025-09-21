use std::arch::x86_64::{
    _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_set1_ps, _mm256_setzero_ps,
};
use std::f32;

use crate::original::array::{max_iter_f32, min_iter_f32};
use crate::x86_64::simd_extensions::{horizontal_max_f32_avx, horizontal_min_f32_avx};

/// Returns the smallest item in the array, or `None` if the array was empty.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn min_avx(array: &[f32]) -> Option<f32> {
    const STEP: usize = 32;

    let len = array.len();
    let m = len % STEP;
    let iterr = len - m;

    let mut i = 0;

    let mut ptr = array.as_ptr();

    let mut lmins1 = _mm256_set1_ps(f32::INFINITY);
    let mut lmins2 = _mm256_set1_ps(f32::INFINITY);
    let mut lmins3 = _mm256_set1_ps(f32::INFINITY);
    let mut lmins4 = _mm256_set1_ps(f32::INFINITY);

    unsafe {
        while i < iterr {
            let current = _mm256_loadu_ps(ptr.cast());
            let current2 = _mm256_loadu_ps(ptr.add(8).cast());
            let current3 = _mm256_loadu_ps(ptr.add(16).cast());
            let current4 = _mm256_loadu_ps(ptr.add(24).cast());

            lmins1 = _mm256_min_ps(current, lmins1);
            lmins2 = _mm256_min_ps(current2, lmins2);
            lmins3 = _mm256_min_ps(current3, lmins3);
            lmins4 = _mm256_min_ps(current4, lmins4);

            i += STEP;
            ptr = ptr.add(STEP);
        }
    }

    let m1 = _mm256_min_ps(lmins1, lmins2);
    let m2 = _mm256_min_ps(lmins3, lmins4);
    let lmins = _mm256_min_ps(m1, m2);
    let min = horizontal_min_f32_avx(lmins);

    if let Some(remainer_min) = min_iter_f32(&array[iterr..]) {
        if min < remainer_min {
            return Some(min);
        } else {
            return Some(remainer_min);
        }
    }

    Some(min)
}

/// Returns the largest item in the array, or `None` if the array was empty.
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
pub fn max_avx(array: &[f32]) -> Option<f32> {
    const STEP: usize = 32;

    let len = array.len();
    let m = len % STEP;
    let iterr = len - m;

    let mut i = 0;

    let mut ptr = array.as_ptr();

    let mut lmax1 = _mm256_setzero_ps();
    let mut lmax2 = _mm256_setzero_ps();
    let mut lmax3 = _mm256_setzero_ps();
    let mut lmax4 = _mm256_setzero_ps();

    unsafe {
        while i < iterr {
            let current = _mm256_loadu_ps(ptr.cast());
            let current2 = _mm256_loadu_ps(ptr.add(8).cast());
            let current3 = _mm256_loadu_ps(ptr.add(16).cast());
            let current4 = _mm256_loadu_ps(ptr.add(24).cast());

            lmax1 = _mm256_max_ps(current, lmax1);
            lmax2 = _mm256_max_ps(current2, lmax2);
            lmax3 = _mm256_max_ps(current3, lmax3);
            lmax4 = _mm256_max_ps(current4, lmax4);

            i += STEP;
            ptr = ptr.add(STEP);
        }
    }

    let m1 = _mm256_max_ps(lmax1, lmax2);
    let m2 = _mm256_max_ps(lmax3, lmax4);
    let lmins = _mm256_max_ps(m1, m2);
    let max = horizontal_max_f32_avx(lmins);

    if let Some(remainer_max) = max_iter_f32(&array[iterr..]) {
        if max > remainer_max {
            return Some(max);
        } else {
            return Some(remainer_max);
        }
    }

    Some(max)
}

#[cfg(test)]
mod test {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::original::array::{max_iter_f32, min_iter_f32};

    pub fn random_array_with_value(
        rng: &mut impl Rng,
        len: usize,
        value: Option<f32>,
        index: Option<usize>,
    ) -> Vec<f32> {
        let mut vec: Vec<_> = (0..len)
            .map(|_| {
                let mut r = rng.next_u32() as f32;

                // If a custom was provided, ensure it's not part of the base array.
                if let Some(val) = value
                    && r == val
                {
                    loop {
                        r = rng.next_u32() as f32;
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
    fn test_array_f32_min_max_fuzzy() {
        let mut rng = StdRng::seed_from_u64(42);

        for len in (32..5000).step_by(13) {
            for _ in 0..10 {
                let vec = random_array_with_value(&mut rng, len, None, None);

                let simd_min = unsafe { min_avx(&vec) };
                let real_min = min_iter_f32(&vec);
                assert_eq!(simd_min, real_min);

                let simd_max = unsafe { max_avx(&vec) };
                let real_max = max_iter_f32(&vec);
                assert_eq!(simd_max, real_max);
            }
        }
    }
}
