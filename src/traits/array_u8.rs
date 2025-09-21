pub trait ArrayU8SimdExt {
    /// Determines the minimum value inside the array.
    fn min_simd(&self) -> Option<u8>;

    /// Determines the maximum value inside the array.
    fn max_simd(&self) -> Option<u8>;

    /// Finds the given `needle` and returns its first occurrence's position or `None` if `needle` is not an element in the array.
    fn find_simd(&self, needle: u8) -> Option<usize>;

    /// Returns `true` if `needle` is an element in the array.
    fn contains_simd(&self, needle: u8) -> bool;

    // /// Counts the occurrences of `element` in the array.
    // fn count_simd(&self, element: u32) -> usize;
}

#[cfg(target_arch = "x86_64")]
impl<T: AsRef<[u8]>> ArrayU8SimdExt for T {
    #[inline]
    fn min_simd(&self) -> Option<u8> {
        use crate::x86_64::AVX2_U8_MIN_SIZE;

        let array = self.as_ref();

        if is_x86_feature_detected!("avx2") && array.len() >= AVX2_U8_MIN_SIZE {
            unsafe { crate::x86_64::array::u8_impl::min_avx(array) }
        } else {
            crate::original::array::min_iter_u8(array)
        }
    }

    #[inline]
    fn max_simd(&self) -> Option<u8> {
        use crate::x86_64::AVX2_U8_MIN_SIZE;
        let array = self.as_ref();

        if is_x86_feature_detected!("avx2") && array.len() >= AVX2_U8_MIN_SIZE {
            unsafe { crate::x86_64::array::u8_impl::max_avx(array) }
        } else {
            crate::original::array::max_iter_u8(array)
        }
    }

    #[inline]
    fn find_simd(&self, needle: u8) -> Option<usize> {
        use crate::x86_64::AVX2_U8_MIN_SIZE;

        let array = self.as_ref();

        if is_x86_feature_detected!("avx2") && array.len() >= AVX2_U8_MIN_SIZE {
            unsafe { crate::x86_64::array::u8_impl::find_avx(array, needle) }
        } else {
            crate::original::array::find_iter_u8(array, needle)
        }
    }

    fn contains_simd(&self, needle: u8) -> bool {
        use crate::x86_64::AVX2_U8_MIN_SIZE;
        let array = self.as_ref();

        if is_x86_feature_detected!("avx2") && array.len() >= AVX2_U8_MIN_SIZE {
            unsafe { crate::x86_64::array::u8_impl::contains_avx(array, needle) }
        } else {
            crate::original::array::contains_iter_u8(array, needle)
        }
    }

    /*
    #[inline]
    fn count_simd(&self, element: u32) -> usize {
        use crate::x86_64::AVX2_U32_MIN_SIZE;
        let array = self.as_ref();

        if is_x86_feature_detected!("avx2") && array.len() >= AVX2_U32_MIN_SIZE {
            unsafe { crate::x86_64::array::u32_impl::count_avx(array, element) }
        } else {
            crate::original::array::count_iter_u32(array, element)
        }
    }*/
}

#[cfg(not(target_arch = "x86_64"))]
impl<T: AsRef<[u8]>> ArrayU8SimdExt for T {
    #[inline]
    fn min_simd(&self) -> Option<u8> {
        crate::original::array::min_iter_u8(self.as_ref())
    }

    #[inline]
    fn max_simd(&self) -> Option<u8> {
        crate::original::array::max_iter_u8(self.as_ref())
    }

    #[inline]
    fn find_simd(&self, needle: u8) -> Option<usize> {
        crate::original::array::find_iter_u8(self.as_ref(), needle)
    }

    #[inline]
    fn contains_simd(&self, needle: u8) -> bool {
        crate::original::array::contains_iter_u8(self.as_ref(), needle)
    }

    // #[inline]
    // fn count_simd(&self, element: u32) -> usize {
    //     crate::original::array::count_iter_u32(self.as_ref(), element)
    // }
}

#[cfg(test)]
mod test {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use crate::original::array::{find_iter_u8, max_iter_u8, min_iter_u8};

    use super::*;

    #[test]
    fn test_array_u32() {
        let mut rng = StdRng::seed_from_u64(42);

        let vec: Vec<u8> = (0..200)
            .map(|_| (rng.next_u32() % u8::MAX as u32) as u8)
            .collect();

        assert!(vec.contains_simd(vec[199]));
        assert_eq!(vec.find_simd(vec[199]), find_iter_u8(&vec, vec[199]));

        // assert_eq!(vec.count_simd(42), count_iter_u32(&vec, 42));

        assert_eq!(vec.max_simd(), max_iter_u8(&vec));
        assert_eq!(vec.min_simd(), min_iter_u8(&vec));
    }
}
