#[inline]
pub(crate) fn find_iter_u32(array: &[u32], needle: u32) -> Option<usize> {
    array.iter().position(|i| *i == needle)
}

#[inline]
pub(crate) fn contains_iter_u32(array: &[u32], needle: u32) -> bool {
    array.iter().any(|i| *i == needle)
}

#[inline]
pub(crate) fn count_iter_u32(array: &[u32], needle: u32) -> usize {
    array.iter().filter(|i| **i == needle).count()
}

#[inline]
pub(crate) fn min_iter_u32(array: &[u32]) -> Option<u32> {
    array.iter().min().copied()
}

#[inline]
pub(crate) fn max_iter_u32(array: &[u32]) -> Option<u32> {
    array.iter().max().copied()
}

#[inline]
pub(crate) fn max_iter_f32(array: &[f32]) -> Option<f32> {
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
pub(crate) fn min_iter_f32(array: &[f32]) -> Option<f32> {
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

#[cfg(test)]
mod test {
    use ordered_float::OrderedFloat;
    use rand::{RngCore, SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn test_min_max_iter_f32() {
        let mut rng = StdRng::seed_from_u64(42);

        for len in (32..5000).step_by(13) {
            for _ in 0..10 {
                let vec: Vec<_> = (0..len).map(|_| rng.next_u32() as f32 / 102.0).collect();

                let our_min = min_iter_f32(&vec);
                let real_min = vec.iter().map(|i| OrderedFloat(*i)).min().map(|i| *i);
                assert_eq!(our_min, real_min);

                let our_max = max_iter_f32(&vec);
                let real_max = vec.iter().map(|i| OrderedFloat(*i)).max().map(|i| *i);
                assert_eq!(our_max, real_max);
            }
        }
    }
}
