pub trait ArrayF32SimdExt {
    /// Determines the minimum value inside the array.
    fn min_simd(&self) -> Option<f32>;

    /// Determines the maximum value inside the array.
    fn max_simd(&self) -> Option<f32>;
}

#[cfg(target_arch = "x86_64")]
impl<T: AsRef<[f32]>> ArrayF32SimdExt for T {
    #[inline]
    fn min_simd(&self) -> Option<f32> {
        use crate::x86_64::AVX2_F32_MIN_SIZE;

        let array = self.as_ref();

        if is_x86_feature_detected!("avx2") && array.len() >= AVX2_F32_MIN_SIZE {
            unsafe { crate::x86_64::array::f32_impl::min_avx(array) }
        } else {
            crate::original::array::min_iter_f32(array)
        }
    }

    #[inline]
    fn max_simd(&self) -> Option<f32> {
        use crate::x86_64::AVX2_F32_MIN_SIZE;

        let array = self.as_ref();

        if is_x86_feature_detected!("avx2") && array.len() >= AVX2_F32_MIN_SIZE {
            unsafe { crate::x86_64::array::f32_impl::max_avx(array) }
        } else {
            crate::original::array::max_iter_f32(array)
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl<T: AsRef<[f32]>> ArrayF32SimdExt for T {
    #[inline]
    fn min_simd(&self) -> Option<f32> {
        crate::original::array::min_iter_f32(self.as_ref())
    }

    #[inline]
    fn max_simd(&self) -> Option<f32> {
        crate::original::array::max_iter_f32(self.as_ref())
    }
}

#[cfg(test)]
mod test {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use super::*;
    use crate::original::array::*;

    #[test]
    fn test_array_u32() {
        let mut rng = StdRng::seed_from_u64(42);

        let vec: Vec<_> = (0..100).map(|_| rng.next_u32() as f32).collect();

        assert_eq!(vec.max_simd(), max_iter_f32(&vec));
        assert_eq!(vec.min_simd(), min_iter_f32(&vec));
    }
}
